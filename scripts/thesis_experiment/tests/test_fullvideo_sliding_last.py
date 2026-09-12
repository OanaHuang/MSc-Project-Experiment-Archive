from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


def _load_evaluator_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "evaluate_fullvideo_subset.py"
    spec = importlib.util.spec_from_file_location("fullvideo_sliding_last", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(video_frames: int, schedule: list[int]) -> dict:
    return {
        "model": {
            "num_steps": len(schedule),
            "temporal": {
                "video_frames": video_frames,
                "update_schedule": schedule,
            },
        },
        "data": {"minimum_temporal_history": video_frames - 1},
        "temporal": {
            "video_frames": video_frames,
            "update_schedule": deepcopy(schedule),
        },
    }


def test_prefix_configs_preserve_causal_update_schedules() -> None:
    evaluator = _load_evaluator_module()
    v4u4 = _config(4, [0, 1, 2, 3])
    assert evaluator._prefix_config(v4u4, 1)["model"]["temporal"]["update_schedule"] == [0]
    assert evaluator._prefix_config(v4u4, 2)["model"]["temporal"]["update_schedule"] == [0, 1]
    assert evaluator._prefix_config(v4u4, 3)["model"]["temporal"]["update_schedule"] == [0, 1, 2]

    v2u4 = _config(2, [0, 0, 1, 1])
    prefix = evaluator._prefix_config(v2u4, 1)
    assert prefix["model"]["num_steps"] == 2
    assert prefix["model"]["temporal"]["update_schedule"] == [0, 0]
    assert prefix["data"]["minimum_temporal_history"] == 0


def test_prefix_selection_keeps_one_warmup_target_per_video() -> None:
    evaluator = _load_evaluator_module()
    dataset = SimpleNamespace(
        samples=[{"sample_id": "a"}, {"sample_id": "b"}],
        frame_index=[
            (0, 0, None, 0), (0, 1, None, 1), (0, 2, None, 2),
            (1, 0, None, 0), (1, 1, None, 1), (1, 2, None, 2),
        ],
    )
    assert evaluator._keep_prefix_target(dataset, 2) == 2
    assert dataset.frame_index == [
        (0, 1, None, 1),
        (1, 1, None, 1),
    ]


def test_prediction_parts_are_merged_into_complete_frame_order() -> None:
    evaluator = _load_evaluator_module()

    def part(frames: list[int]) -> dict[str, np.ndarray]:
        count = len(frames)
        return {
            "prediction": np.asarray(frames, dtype=np.float32)[:, None],
            "target": np.asarray(frames, dtype=np.float32)[:, None],
            "visibility": np.ones((count, 1), dtype=np.float32),
            "scale": np.ones(count, dtype=np.float32),
            "scale_hb": np.ones(count, dtype=np.float32),
            "sample_id": np.asarray(["video"] * count),
            "frame_index": np.asarray(frames),
            "person_id": np.asarray(["primary"] * count),
            "video_id": np.asarray(["video"] * count),
            "clip_id": np.asarray(["full"] * count),
            "frame_position_in_clip": np.asarray(frames),
        }

    merged = evaluator._merge_prediction_values([
        part([3, 4, 5]),
        part([0]),
        part([1]),
        part([2]),
    ])
    assert merged["frame_index"].tolist() == [0, 1, 2, 3, 4, 5]
    assert merged["prediction"][:, 0].tolist() == [0, 1, 2, 3, 4, 5]
