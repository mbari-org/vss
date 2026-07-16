# fastapi-vss, Apache-2.0 license
# Filename: predictors/process_vits.py
# Description: Process images with Vision Transformer (ViT) model and search by KNN embeddings in Redis vector store
import os

import numpy as np
import redis  # type: ignore
import torch
from PIL import Image, ImageFile  # type: ignore
from transformers import AutoModel, AutoImageProcessor  # type: ignore
from typing import Iterator, List, Tuple

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


class ViTWrapper:
    def __init__(self, r: redis.Redis, device, model_name: str, reset: bool = False, batch_size: int = 32):
        self.r = r
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(model_name)
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
        if os.getenv("VSS_TORCH_COMPILE", "0") == "1":
            info("Compiling model with torch.compile")
            self.model = torch.compile(self.model)

        if model_name.startswith("/"):
            if not os.path.exists(model_name):
                raise FileNotFoundError(f"Model directory {model_name} does not exist")

    @property
    def vector_dimensions(self) -> int:
        return self.model.config.hidden_size

    def preprocess_images(self, image_paths: List[str]) -> Iterator[Tuple[dict, List[str], List[str]]]:
        debug(f"Preprocessing {len(image_paths)} images")
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = []
            valid_paths = []
            failed_paths = []
            for p in batch_paths:
                try:
                    img = Image.open(p)
                    img.load()
                    images.append(img.convert("RGB"))
                    valid_paths.append(p)
                except Exception as e:
                    logger.warning(f"Skipping unreadable image {p}: {e}")
                    failed_paths.append(p)
            if not images:
                logger.warning(f"No valid images in batch starting at index {i}, skipping")
                continue
            if failed_paths:
                logger.warning(f"Batch reduced from {len(batch_paths)} to {len(images)} images due to read errors")
            inputs = self.processor(images=images, return_tensors="pt")
            debug(f"Done preprocessing batch of {len(images)} images")
            yield inputs, valid_paths, failed_paths

    def get_image_embeddings(self, inputs):
        """get embeddings for a batch of images"""
        debug(f"Getting embeddings for batch of size {inputs['pixel_values'].shape[0]}")
        debug(inputs["pixel_values"].shape)  # Should be (B, 3, H, W)

        # Move inputs to same device as model (processor returns CPU tensors)
        inputs = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.inference_mode():
            if self.amp_dtype is not None:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    embeddings = self.model(**inputs)
            else:
                embeddings = self.model(**inputs)
        info("Done getting embeddings for batch")
        # Cast back to float32 to match the Redis index (FLOAT32) regardless of autocast dtype
        batch_embeddings = embeddings.last_hidden_state[:, 0, :].float().cpu().numpy()
        info(f"Batch embeddings shape: {batch_embeddings.shape}")
        return np.array(batch_embeddings)

    def predict(self, image_paths: List[str], top_n: int = 1) -> tuple[list[list[str]], list[list[float]], list[list[str]]]:
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

    def get_embeddings(self, image_paths: List[str], filenames: List[str] | None = None) -> Tuple[List[List[float]], List[str]]:
        """Get embeddings for a batch of images. Returns (embeddings, failed_filenames)."""
        all_embeddings = []
        failed_paths: List[str] = []
        path_to_filename = dict(zip(image_paths, filenames)) if filenames else {}

        info(f"Found {len(image_paths)} images to get embeddings")
        for inputs, valid_paths, batch_failed_paths in self.preprocess_images(image_paths):
            failed_paths.extend(batch_failed_paths)
            embeddings = self.get_image_embeddings(inputs)
            for emb in embeddings:
                all_embeddings.append(emb.tolist())

        failed_filenames = [path_to_filename.get(p, p) for p in failed_paths]
        return all_embeddings, failed_filenames
