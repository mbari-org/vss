# fastapi-vss, Apache-2.0 license
# Filename: predictors/process_vits.py
# Description: Process images with Vision Transformer (ViT) model and search by KNN embeddings in Redis vector store
import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import redis  # type: ignore
import torch
from PIL import Image, ImageFile  # type: ignore
from transformers import AutoModel, AutoImageProcessor  # type: ignore
from typing import Iterator, List, Optional, Tuple, Union

from app.predictors.vector_similarity import VectorSimilarity

import logging

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.DEBUG)
debug = logging.debug
info = logging.info
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(console)


# An image to embed: a filesystem path, or an in-memory buffer of encoded bytes.
ImageSource = Union[str, bytes, io.BytesIO]


def _default_decode_workers() -> int:
    """Threads used to decode one batch (VSS_DECODE_WORKERS, default min(8, cpu count))."""
    raw = os.getenv("VSS_DECODE_WORKERS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning(f"Ignoring invalid VSS_DECODE_WORKERS={raw!r}")
    return max(1, min(8, os.cpu_count() or 1))


def _open_rgb(source: ImageSource) -> Image.Image:
    """Decode one image from a path or an in-memory buffer into RGB."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    elif isinstance(source, io.IOBase):
        source.seek(0)
    img = Image.open(source)
    # Force the decode here rather than lazily later: this call is what the thread pool is
    # parallelizing, and PIL releases the GIL inside it.
    img.load()
    return img.convert("RGB")


class ViTWrapper:
    def __init__(self, r: redis.Redis, device, model_name: str, reset: bool = False, batch_size: int = 32):
        self.r = r
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.batch_size = batch_size
        # transformers resolves a *slow* image processor when `use_fast` is unset, unless the
        # checkpoint was saved with a fast one -- in 4.57 only Qwen2VL is force-upgraded
        # (FORCE_FAST_IMAGE_PROCESSOR). The slow path resizes, rescales and normalizes one
        # image at a time in Python/NumPy on the CPU, and it dominated batch latency: seconds
        # per batch of 128 crops, all of it with the GPU idle. The torchvision-backed fast
        # processor does the same work batched.
        #
        # Fast output differs very slightly (torchvision resize vs PIL), so embeddings shift
        # a little; set VSS_FAST_PROCESSOR=0 to keep the old behaviour when querying a Redis
        # index that was built with the slow processor.
        want_fast = os.getenv("VSS_FAST_PROCESSOR", "1") == "1"
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=want_fast)
        except Exception as e:
            logger.warning(
                f"Could not load image processor with use_fast={want_fast} ({e}); "
                "falling back to the default processor"
            )
            self.processor = AutoImageProcessor.from_pretrained(model_name)
        info(f"Image processor: {type(self.processor).__name__} (requested use_fast={want_fast})")

        self.decode_workers = _default_decode_workers()
        info(f"Image decode threads: {self.decode_workers}")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        # eval() disables dropout and puts the model in inference mode
        self.model.eval()

        # Mixed-precision (fp16) autocast roughly halves memory and ~2x GPU throughput.
        # Only enabled on CUDA; can be disabled with VSS_AMP=0.
        amp_enabled = os.getenv("VSS_AMP", "1") == "1"
        self.amp_dtype = torch.float16 if (amp_enabled and self.device.type == "cuda") else None

        self.vs = VectorSimilarity(r, vector_dimensions=self.vector_dimensions, reset=reset)

        # Optional torch.compile for a further speedup (one-time warmup cost).
        # Done last so vector_dimensions is read from the original (uncompiled) module.
        # Compilation is lazy (happens on the first forward), and its Inductor/Triton
        # backend requires a C compiler + CUDA headers in the runtime image. If those
        # are missing the first forward raises; we keep an eager reference so we can
        # fall back gracefully instead of failing every prediction (see get_image_embeddings).
        self._compiled = False
        if os.getenv("VSS_TORCH_COMPILE", "0") == "1":
            info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)
            self._compiled = True

        if model_name.startswith("/"):
            if not os.path.exists(model_name):
                raise FileNotFoundError(f"Model directory {model_name} does not exist")

    @property
    def vector_dimensions(self) -> int:
        return self.model.config.hidden_size

    def preprocess_images(
        self,
        image_sources: List[ImageSource],
        labels: Optional[List[str]] = None,
    ) -> Iterator[Tuple[dict, List[str], List[str]]]:
        """
        Decode images and run the processor, yielding one model-ready batch at a time.

        `image_sources` may be filesystem paths or in-memory buffers of encoded bytes.
        `labels` names each source for logging and for the failed-image report; it defaults
        to the source itself, so passing a list of paths behaves exactly as before.

        Yields (inputs, valid_labels, failed_labels).
        """
        debug(f"Preprocessing {len(image_sources)} images")
        names = list(labels) if labels else [str(s) for s in image_sources]

        def _safe_open(item: Tuple[ImageSource, str]) -> Optional[Image.Image]:
            source, name = item
            try:
                return _open_rgb(source)
            except Exception as e:
                logger.warning(f"Skipping unreadable image {name}: {e}")
                return None

        for i in range(0, len(image_sources), self.batch_size):
            batch = image_sources[i : i + self.batch_size]
            batch_names = names[i : i + self.batch_size]

            # Decode on a thread pool. PIL releases the GIL inside load(), so this scales
            # with cores; done serially it cost seconds per batch of large crops, with the
            # GPU sitting idle throughout. pool.map preserves order, which matters because
            # embeddings are matched back to their inputs positionally.
            workers = max(1, min(self.decode_workers, len(batch)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                decoded = list(pool.map(_safe_open, zip(batch, batch_names)))

            images = []
            valid_labels: List[str] = []
            failed_labels: List[str] = []
            for name, img in zip(batch_names, decoded):
                if img is None:
                    failed_labels.append(name)
                else:
                    images.append(img)
                    valid_labels.append(name)

            if not images:
                logger.warning(f"No valid images in batch starting at index {i}, skipping")
                continue
            if failed_labels:
                logger.warning(f"Batch reduced from {len(batch)} to {len(images)} images due to read errors")
            inputs = self.processor(images=images, return_tensors="pt")
            debug(f"Done preprocessing batch of {len(images)} images")
            yield inputs, valid_labels, failed_labels

    def _forward(self, inputs):
        """Run the model forward pass under inference_mode and optional fp16 autocast."""
        with torch.inference_mode():
            if self.amp_dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    return self.model(**inputs)
            return self.model(**inputs)

    def get_image_embeddings(self, inputs):
        """get embeddings for a batch of images"""
        debug(f"Getting embeddings for batch of size {inputs['pixel_values'].shape[0]}")
        debug(inputs["pixel_values"].shape)  # Should be (B, 3, H, W)

        # Move inputs to same device as model (processor returns CPU tensors)
        inputs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        try:
            embeddings = self._forward(inputs)
        except Exception as e:
            # torch.compile is lazy: Inductor/Triton kernel compilation happens on the
            # first forward and can fail if the runtime lacks a C compiler/CUDA headers.
            # Fall back to eager execution once instead of failing every prediction.
            if self._compiled:
                logger.warning(f"torch.compile forward failed ({e}); falling back to eager execution")
                self.model = getattr(self.model, "_orig_mod", self.model)
                self._compiled = False
                embeddings = self._forward(inputs)
            else:
                raise
        info("Done getting embeddings for batch")
        # Cast back to float32 to match the Redis index (FLOAT32) regardless of autocast dtype
        batch_embeddings = embeddings.last_hidden_state[:, 0, :].float().cpu().numpy()
        info(f"Batch embeddings shape: {batch_embeddings.shape}")
        return np.array(batch_embeddings)

    def predict(self, image_paths: List[ImageSource], top_n: int = 1) -> tuple[list[list[str]], list[list[float]], list[list[str]]]:
        """Search using KNN for embeddings for a batch of images"""
        predictions = []
        scores = []
        ids = []

        info(f"Found {len(image_paths)} images to predict")
        for inputs, _, _ in self.preprocess_images(image_paths):
            embeddings = self.get_image_embeddings(inputs)
            info(f"Searching for {len(embeddings)} embeddings in Redis")
            # Issue all KNN searches for the batch concurrently instead of one round-trip per image
            vectors = [emb.tobytes() for emb in embeddings]
            results = self.vs.search_vectors(vectors, top_n=top_n)
            for r in results:
                # Data is doc:label:id - split it into parts
                data = [x["id"].split(":") for x in r]
                batch_pred = []
                batch_ids = []
                for d in data:
                    batch_pred.append(d[1])
                    batch_ids.append(d[2])

                predictions.append([b for b in batch_pred])
                ids.append([i for i in batch_ids])
                # Separate out the scores for each prediction - this is used later for voting
                scores.append([round(float(x["score"]), 4) for x in r])

        return predictions, scores, ids

    def get_embeddings(self, image_sources: List[ImageSource], filenames: List[str] | None = None) -> Tuple[List[List[float]], List[str]]:
        """Get embeddings for a batch of images. Returns (embeddings, failed_filenames)."""
        all_embeddings = []
        failed_filenames: List[str] = []

        info(f"Found {len(image_sources)} images to get embeddings")
        # Labels are the caller's filenames, so failures come back named without having to
        # key a dict on the sources themselves (which may be anonymous in-memory buffers).
        for inputs, _, batch_failed in self.preprocess_images(image_sources, labels=filenames):
            failed_filenames.extend(batch_failed)
            embeddings = self.get_image_embeddings(inputs)
            for emb in embeddings:
                all_embeddings.append(emb.tolist())

        return all_embeddings, failed_filenames
