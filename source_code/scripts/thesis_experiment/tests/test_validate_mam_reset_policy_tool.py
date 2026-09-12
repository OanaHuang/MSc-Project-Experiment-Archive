from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch


TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "validate_mam_reset_policy.py"
)
SPEC = importlib.util.spec_from_file_location("validate_mam_reset_policy_tool", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class _SpatialStub:
    def forward_spatial_per_step(self, image: torch.Tensor) -> torch.Tensor:
        return image.new_zeros((image.shape[1], 1, 1, 1, 1))


def test_overlapping_invisible_nan_targets_are_identical_metadata() -> None:
    batch = {
        "image": torch.zeros(2, 1, 3, 4, 4),
        "temporal_frame_positions": torch.zeros(2, 1, dtype=torch.int64),
        "temporal_frame_indices": torch.full((2, 1), 83, dtype=torch.int64),
        "temporal_keypoints": torch.full((2, 1, 1, 2), float("nan")),
        "temporal_visibility": torch.zeros(2, 1, 1),
        "person_bbox": torch.tensor([
            [606.0, 212.0, 1364.0, 970.0],
            [606.0, 212.0, 1364.0, 970.0],
        ]),
        "sample_id": ["S004C002P020R001A021"] * 2,
        "video_id": ["S004C002P020R001A021"] * 2,
        "person_id": ["72057594037933363"] * 2,
    }
    records = {}
    TOOL._add_batch_records(records, batch, _SpatialStub(), torch.device("cpu"))
    assert list(records) == [0]
    assert np.isnan(records[0]["target"]).all()
