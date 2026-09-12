from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

from spikepose_thesis.core.config import load_experiment, load_study
from spikepose_thesis.data.factory import build_dataset
from spikepose_thesis.data.mpii.official_protocol import (
    EXPECTED_ANNOTATION_SHA256, prepare_official_mpii_protocol,
)
from spikepose_thesis.experiments.readiness import experiment_readiness


def _json_hash(value) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_official_backbone_study_has_seven_table_rows():
    study = load_study("mpii_official_backbones")
    assert study["experiments"] == [
        "mpii_official_spikepose_ann",
        "mpii_official_resnet50",
        "mpii_official_hrnet_w32",
        "mpii_official_spikeyolo",
        "mpii_official_spikformer",
        "mpii_official_spikepose_u1",
        "mpii_official_spikepose_u2",
    ]
    for name in study["experiments"]:
        config = load_experiment(name)
        assert config["data"]["split_protocol"] == "official_hrnet_mpii"
        assert config["data"]["validation_metadata"].endswith(
            "metadata/official_hrnet/val.jsonl"
        )


def test_released_baselines_are_evaluation_only():
    for name, family, rgb in (
        ("mpii_official_resnet50", "pose_resnet", False),
        ("mpii_official_hrnet_w32", "hrnet", True),
    ):
        config = load_experiment(name)
        assert config["action"] == "evaluate_official_checkpoint"
        assert config["training"]["epochs"] == 0
        assert config["model"]["family"] == family
        assert config["data"]["color_rgb"] is rgb


def test_spikeyolo_uses_the_shared_linear_heatmap_head():
    config = load_experiment("mpii_official_spikeyolo")
    assert config["paper_id"] == "MPII-Official-SpikeYOLO"
    assert config["initialization"] == {"mode": "scratch"}
    assert config["model"]["head"]["kind"] == "linear_heatmap"
    assert config["model"]["backbone"]["feature_layer"] == 19
    assert config["model"]["backbone"]["feature_channels"] == 128
    assert config["implementation_status"] == "ready"
    assert config["model"]["backbone"]["keep_layers"] == [0, 19]


def test_prepare_and_load_official_protocol(tmp_path, monkeypatch):
    official = tmp_path / "official"
    legacy = tmp_path / "legacy"
    output = tmp_path / "prepared"
    images = tmp_path / "images"
    for path in (official, legacy, images):
        path.mkdir()
    row = {
        "image": "sample.jpg", "center": [20.0, 20.0], "scale": 1.0,
        "joints": [[10.0, 10.0]] * 16, "joints_vis": [1] * 16,
    }
    for name in ("train.json", "valid.json"):
        value = [row]
        (official / name).write_text(
            json.dumps(value, separators=(",", ":")), encoding="utf-8",
        )
        monkeypatch.setitem(EXPECTED_ANNOTATION_SHA256, name, _json_hash(value))
    legacy_row = {
        "image": "sample.jpg", "person_index": 0,
        "center": [20.0, 20.0], "scale": 1.0,
        "keypoints": [[10.0, 10.0]] * 16, "visibility": [1] * 16,
        "head_length": 5.0,
    }
    for name in ("train.jsonl", "val.jsonl"):
        (legacy / name).write_text(json.dumps(legacy_row) + "\n", encoding="utf-8")
    cv2.imwrite(str(images / "sample.jpg"), np.zeros((40, 40, 3), np.uint8))

    report = prepare_official_mpii_protocol(official, legacy, output, images)
    assert report["splits"]["train.jsonl"]["prepared_records"] == 1
    assert report["splits"]["val.jsonl"]["prepared_records"] == 1
    config = load_experiment("mpii_official_spikepose_u1")
    config["data"].update({
        "images_dir": str(images),
        "train_metadata": str(output / "train.jsonl"),
        "validation_metadata": str(output / "val.jsonl"),
    })
    dataset = build_dataset(config, "validation")
    sample = dataset[0]
    assert len(dataset) == 1
    assert sample["image"].shape == (3, 256, 256)
    assert sample["heatmaps"].shape == (16, 64, 64)
    assert np.array_equal(sample["inverse"], sample["inverse_affine"])
