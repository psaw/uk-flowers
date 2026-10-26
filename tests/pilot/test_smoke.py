"""Offline integration checks for the pilot execution environment."""

import importlib
import json
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
import pytorch_lightning as pl
import torch

from uk_flowers.data.dataset import TestDataset as FlowerTestDataset
from uk_flowers.model.module import FlowerClassifier


def test_dataset_reads_local_image(tmp_path: Path) -> None:
    image_path = tmp_path / "flower.jpg"
    Image.new("RGB", (32, 32), color="red").save(image_path)
    dataset = FlowerTestDataset(tmp_path)
    assert len(dataset) == 1
    image, label, name = dataset[0]
    assert image.mode == "RGB"
    assert label == 0
    assert name == "flower.jpg"


def test_inference_with_local_checkpoint(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    model = FlowerClassifier(pretrained=False, num_classes=2)
    checkpoint = tmp_path / "model.ckpt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyper_parameters": dict(model.hparams),
            "pytorch-lightning_version": pl.__version__,
        },
        checkpoint,
    )
    image_path = tmp_path / "flower.jpg"
    Image.new("RGB", (256, 256), color="yellow").save(image_path)
    (tmp_path / "cat_to_name.json").write_text(
        json.dumps({"1": "class-one", "2": "class-two"}), encoding="utf-8"
    )
    output = tmp_path / "result.json"
    cfg = OmegaConf.create(
        {
            "project": {"seed": 42, "device": "cpu"},
            "data": {"data_dir": str(tmp_path)},
            "inference": {
                "image_path": str(image_path),
                "checkpoint_path": str(checkpoint),
                "output_path": str(output),
            },
        }
    )
    infer_module = importlib.import_module("uk_flowers.infer")
    infer_module.infer(cfg)
    results = json.loads(output.read_text(encoding="utf-8"))
    assert set(results) == {str(image_path)}
    prediction = results[str(image_path)]
    assert prediction["class_idx"] in {1, 2}
    assert prediction["class_name"] in {"class-one", "class-two"}
    assert 0 <= prediction["probability"] <= 1
