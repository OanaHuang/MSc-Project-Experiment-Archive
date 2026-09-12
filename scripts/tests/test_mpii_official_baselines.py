from __future__ import annotations

import json
import hashlib

import cv2
import numpy as np
import torch

from scripts.MPII.core.official_transforms import affine_matrix, transform_points
from scripts.MPII.datasets import OfficialMPIIDataset
from scripts.MPII.evaluate_official_baseline import load_checkpoint
from scripts.MPII.prepare_official_protocol import prepare
from scripts.spikepose.models.baselines import build_hrnet, build_pose_resnet


def test_checkpoint_tensor_fingerprint(tmp_path):
    model = torch.nn.Linear(2, 3)
    checkpoint = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint)
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    assert load_checkpoint(torch.nn.Linear(2, 3), checkpoint) == digest.hexdigest()


def test_official_models_emit_mpii_heatmaps():
    image = torch.zeros(1, 3, 256, 256)
    for model in (build_pose_resnet(50), build_hrnet(32), build_hrnet(48)):
        model.eval()
        with torch.no_grad():
            output = model(image)
        assert output.shape == (1, 16, 64, 64)


def test_official_affine_round_trip():
    points = np.asarray([[10.0, 20.0], [150.0, 80.0]], np.float32)
    forward = affine_matrix([100, 100], [1.2, 1.2], 0, [256, 256])
    inverse = affine_matrix([100, 100], [1.2, 1.2], 0, [256, 256], inverse=True)
    restored = transform_points(transform_points(points, forward), inverse)
    np.testing.assert_allclose(restored, points, atol=1e-4)


def test_prepare_and_load_official_protocol(tmp_path):
    official, legacy, output, images = [tmp_path / name for name in
                                         ("official", "legacy", "output", "images")]
    for path in (official, legacy, images):
        path.mkdir()
    row = {
        "image": "sample.jpg", "center": [32.0, 32.0], "scale": 0.32,
        "joints": [[20.0, 20.0]] * 16, "joints_vis": [1] * 16,
    }
    for name in ("train.json", "valid.json"):
        (official / name).write_text(json.dumps([row]), encoding="utf-8")
    legacy_row = {
        "image": "sample.jpg", "center": [32.0, 32.0], "scale": 0.32,
        "keypoints": row["joints"], "visibility": row["joints_vis"],
        "head_length": 10.0, "person_index": 0,
    }
    for name in ("train.jsonl", "val.jsonl"):
        (legacy / name).write_text(json.dumps(legacy_row) + "\n", encoding="utf-8")
    cv2.imwrite(str(images / "sample.jpg"), np.zeros((64, 64, 3), np.uint8))
    report = prepare(official, legacy, output, images)
    assert report["splits"]["val.jsonl"]["prepared_records"] == 1
    assert report["splits"]["train.jsonl"]["fallback_records"] == 0
    dataset = OfficialMPIIDataset(output / "val.jsonl", images)
    sample = dataset[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["heatmaps"].shape == (16, 64, 64)
    bgr_dataset = OfficialMPIIDataset(output / "val.jsonl", images, color_rgb=False)
    assert not torch.equal(sample["image"], bgr_dataset[0]["image"])
