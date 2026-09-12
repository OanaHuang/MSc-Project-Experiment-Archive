"""Queue launch safety and validation-selection checks without CUDA jobs."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def queue(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("run_mam_softmax_fix_queue")


@pytest.mark.parametrize("exit_code", [0, 1])
def test_busy_gpu_and_failure_stop(queue, monkeypatch, tmp_path, exit_code):
    launched = []

    def launch(command, **kwargs):
        launched.append((command, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
        return SimpleNamespace(pid=123, poll=lambda: exit_code)

    monkeypatch.setattr(queue.subprocess, "Popen", launch)
    monkeypatch.setattr(queue, "_gpu_occupancy", lambda: ({0, 1}, {0}))
    monkeypatch.setattr(queue.time, "sleep", lambda _: None)
    args = SimpleNamespace(output=tmp_path, gpus=[0, 1], poll_seconds=.01)
    state = {"history": []}
    jobs = [{"name": str(i), "command": ["fake", str(i)]} for i in range(2)]
    if exit_code:
        with pytest.raises(RuntimeError, match="failed jobs"):
            queue.run_jobs(jobs, args, state)
        assert len(launched) == 1
        assert state["pending"] == ["1"]
    else:
        queue.run_jobs(jobs, args, state)
        assert len(launched) == 2
    assert all(gpu == "1" for _, gpu in launched)


def test_pck_constraint_before_temporal_ranking(queue, tmp_path):
    for name, pck, acceleration in [("softmax", .9, 10), ("dark", .89, 1), ("relu", .899, 5)]:
        path = tmp_path / "evaluation" / ("mam_softmax_fix_" + name)
        path.mkdir(parents=True)
        (path / "summary.json").write_text(json.dumps({"scores": {"trained": {
            "pckhb": pck, "mpjacce": acceleration, "mpjve": 2}}}))
    result = queue.select_ablation_candidate(tmp_path, ["dark", "relu"])
    assert result["winner"] == "relu"
    assert result["eligible"] == ["relu"]


def test_paired_comparison_requires_identical_videos(queue, tmp_path):
    for label, video in [("softmax", "v1"), ("dark", "v2")]:
        path = tmp_path / "evaluation" / ("mam_softmax_fix_" + label)
        path.mkdir(parents=True)
        (path / "summary.json").write_text(json.dumps({"per_clip": [{
            "mode": "trained", "video": video, "pck_correct": 1, "pck_valid": 2,
            "velocity_sum": 3, "velocity_n": 2, "acceleration_sum": 4, "acceleration_n": 1}]}))
    with pytest.raises(RuntimeError, match="identical video"):
        queue.compare_results(tmp_path, ["softmax", "dark"], "dark")
