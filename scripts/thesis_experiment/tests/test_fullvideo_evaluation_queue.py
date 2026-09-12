from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_queue_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_fullvideo_evaluation_queue.py"
    spec = importlib.util.spec_from_file_location("fullvideo_queue", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remaining_paper_jobs_are_unique_and_complete() -> None:
    queue = _load_queue_module()
    labels = [job.label for job in queue.MODELS]
    assert len(labels) == len(set(labels))
    assert set(labels) == {
        "simplebaseline_r50", "hrnet_w32", "spikepose_ann",
        "spikepose_ann_mam", "spikeyolo", "v1u1", "v1u2", "v1u4",
        "v2u2", "v2u4", "v4u4",
    }
    assert next(
        job for job in queue.MODELS if job.label == "spikepose_ann_mam"
    ).evaluator == "mam_reset"


def test_queue_can_select_only_failed_labels() -> None:
    queue = _load_queue_module()
    selected = queue._select_models("v1u1,v2u4,v4u4")
    assert [job.label for job in selected] == ["v1u1", "v2u4", "v4u4"]


def test_queue_rejects_unknown_or_duplicate_labels() -> None:
    queue = _load_queue_module()
    with pytest.raises(ValueError, match="Unknown model labels"):
        queue._select_models("v1u1,missing")
    with pytest.raises(ValueError, match="must not be repeated"):
        queue._select_models("v1u1,v1u1")
