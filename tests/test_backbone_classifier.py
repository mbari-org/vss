"""BackboneClassifier checkpoints (DINOv3 fine-tunes) are not an HF architecture."""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModel

from app.predictors.load_model import load_vision_model


def _tiny_backbone():
    candidates = (
        (
            "dinov3_vit",
            dict(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=64,
                image_size=16,
                patch_size=16,
                num_register_tokens=4,
            ),
        ),
        (
            "vit",
            dict(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                intermediate_size=64,
                image_size=16,
                patch_size=16,
            ),
        ),
    )
    for model_type, kwargs in candidates:
        try:
            config = AutoConfig.for_model(model_type, **kwargs)
            return AutoModel.from_config(config)
        except Exception:
            continue
    pytest.skip("transformers has neither dinov3_vit nor vit")


def _write_backbone_classifier(tmp_path: Path, backbone) -> None:
    backbone_config = json.loads(backbone.config.to_json_string())
    config = {
        "architectures": ["BackboneClassifier"],
        "model_type": "backbone_classifier",
        "base_model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "hidden_size": backbone.config.hidden_size,
        "num_labels": 3,
        "id2label": {"0": "a", "1": "b", "2": "c"},
        "label2id": {"a": 0, "b": 1, "c": 2},
        "backbone_config": backbone_config,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    state = {f"backbone.{k}": v for k, v in backbone.state_dict().items()}
    state["classifier.weight"] = torch.zeros(3, backbone.config.hidden_size)
    state["classifier.bias"] = torch.zeros(3)
    try:
        from safetensors.torch import save_file

        save_file(state, (tmp_path / "model.safetensors").as_posix())
    except ImportError:
        torch.save(state, tmp_path / "pytorch_model.bin")


def test_load_backbone_classifier_keeps_finetuned_weights(tmp_path):
    backbone = _tiny_backbone()
    with torch.no_grad():
        for param in backbone.parameters():
            param.add_(0.1)
    expected = {k: v.detach().clone() for k, v in backbone.state_dict().items()}
    _write_backbone_classifier(tmp_path, backbone)

    with pytest.raises(ValueError, match="backbone_classifier"):
        AutoModel.from_pretrained(tmp_path)

    loaded = load_vision_model(str(tmp_path))
    assert type(loaded).__name__ == type(backbone).__name__
    assert loaded.config.hidden_size == backbone.config.hidden_size
    for key, value in expected.items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_backbone_classifier_without_architecture_is_rejected(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["BackboneClassifier"], "model_type": "backbone_classifier"}))
    torch.save({"classifier.weight": torch.zeros(1, 1)}, tmp_path / "pytorch_model.bin")

    with pytest.raises(ValueError, match="neither backbone_config nor base_model"):
        load_vision_model(str(tmp_path))
