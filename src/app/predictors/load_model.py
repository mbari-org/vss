# fastapi-vss, Apache-2.0 license
# Filename: predictors/load_model.py
# Description: Load a vision backbone, including BackboneClassifier fine-tunes
import json
import logging
import os

import torch
from transformers import AutoConfig, AutoModel  # type: ignore

logger = logging.getLogger(__name__)


def _is_backbone_classifier(config: dict) -> bool:
    """True for a fine-tune saved as a plain classification wrapper, not an HF architecture.

    DINOv3 has no AutoModelForImageClassification class, so those runs save a
    BackboneClassifier: config.json says model_type "backbone_classifier" and the
    weights are stored under a "backbone." prefix. AutoModel cannot resolve that type.
    """
    if config.get("model_type") == "backbone_classifier":
        return True
    return "BackboneClassifier" in (config.get("architectures") or [])


def _read_state_dict(model_dir: str) -> dict:
    safe = os.path.join(model_dir, "model.safetensors")
    binary = os.path.join(model_dir, "pytorch_model.bin")
    if os.path.isfile(safe):
        from safetensors.torch import load_file

        return load_file(safe)
    if os.path.isfile(binary):
        return torch.load(binary, map_location="cpu", weights_only=True)
    raise ValueError(f"No model.safetensors or pytorch_model.bin in {model_dir}")


def _load_backbone_classifier(model_dir: str, config: dict):
    """Rebuild the fine-tuned backbone and drop the classification head.

    The head is unused here: embeddings are the backbone CLS token. Leaving the
    wrapper's model_type in place makes AutoModel raise, and loading the state dict
    without stripping the "backbone." prefix would keep randomly initialized weights.
    """
    backbone_config = config.get("backbone_config")
    if backbone_config:
        backbone_config = dict(backbone_config)
        model_type = backbone_config.pop("model_type", None)
        if not model_type:
            raise ValueError(f"backbone_config in {model_dir}/config.json has no model_type")
        model = AutoModel.from_config(AutoConfig.for_model(model_type, **backbone_config))
        logger.info(f"Rebuilt a {model_type} backbone from {model_dir}")
    elif config.get("base_model"):
        model = AutoModel.from_pretrained(config["base_model"])
        logger.info(f"Rebuilt the backbone from {config['base_model']}")
    else:
        raise ValueError(f"{model_dir}/config.json has neither backbone_config nor base_model, so the architecture cannot be reconstructed.")

    state = _read_state_dict(model_dir)
    backbone_state = {}
    foreign = []
    for key, value in state.items():
        if key.startswith("backbone."):
            backbone_state[key[len("backbone.") :]] = value
        elif key.startswith(("classifier.", "dropout.")):
            continue
        else:
            foreign.append(key)

    if not backbone_state:
        raise ValueError(f"No backbone weights in {model_dir}; keys start with {sorted({k.split('.')[0] for k in state})}")
    if foreign:
        raise ValueError(f"Unexpected weight keys in {model_dir}: {foreign[:8]}")

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    if missing or unexpected:
        raise ValueError(f"Backbone weights in {model_dir} do not match the architecture in backbone_config (missing={len(missing)}, unexpected={len(unexpected)}).")
    return model


def load_vision_model(model_name: str):
    """Load a vision backbone, including BackboneClassifier fine-tunes AutoModel cannot open."""
    config_path = os.path.join(model_name, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        if _is_backbone_classifier(config):
            logger.info(f"{model_name} is a BackboneClassifier save; loading its backbone and dropping the head")
            return _load_backbone_classifier(model_name, config)
    return AutoModel.from_pretrained(model_name)
