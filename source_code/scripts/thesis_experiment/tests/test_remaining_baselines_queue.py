import json
import subprocess
import sys

import pytest

from tools import run_remaining_baselines_queue as queue


def setup_queue(monkeypatch, tmp_path, readiness="RESUMABLE"):
    name = "mpii_official_spikepose_u2"
    run = tmp_path / "run"
    run.mkdir()
    monkeypatch.setattr(queue, "EXPERIMENTS", [name])
    monkeypatch.setattr(queue, "audit_data", lambda: {})
    monkeypatch.setattr(queue, "load_experiment", lambda _: {"dataset": "mpii"})
    monkeypatch.setattr(queue, "experiment_readiness", lambda *a, **k: {"status": readiness})
    monkeypatch.setattr(queue, "run_dir", lambda *a: run)
    monkeypatch.setattr(queue.subprocess, "check_output", lambda *a, **k: "test-commit")
    monkeypatch.setattr(sys, "argv", ["queue", "--gpu", "2", "--queue-dir", str(tmp_path / "queue")])
    return name, run


def test_queue_resumes_then_evaluates(monkeypatch, tmp_path):
    name, run = setup_queue(monkeypatch, tmp_path)
    commands = []
    def execute(command, **kwargs):
        commands.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
        (run / "status.json").write_text(json.dumps({"state": "completed"}))
    monkeypatch.setattr(queue.subprocess, "run", execute)
    queue.main()
    assert "--resume" in commands[0]
    assert commands[0][4] == "train"
    assert commands[1][4] == "evaluate"
    state = json.loads((tmp_path / "queue/queue.json").read_text())
    assert state[name]["state"] == "completed"


def test_queue_stops_on_failure(monkeypatch, tmp_path):
    name, _ = setup_queue(monkeypatch, tmp_path)
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(queue.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        queue.main()
    state = json.loads((tmp_path / "queue/queue.json").read_text())
    assert state[name]["state"] == "failed"


def test_queue_reports_unmet_dependency(monkeypatch, tmp_path):
    setup_queue(monkeypatch, tmp_path, "WAITING")
    with pytest.raises(RuntimeError, match="blocked"):
        queue.main()
