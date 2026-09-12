import csv
import json

import cv2
import torch

from spikepose_thesis.artifacts import run_dir
from spikepose_thesis.core.config import (
    apply_pilot_runtime_overrides,
    load_experiment,
    parse_ntu_setups,
)
from spikepose_thesis.experiments.readiness import experiment_readiness
from spikepose_thesis.training.checkpoint import (
    restore_training_state,
    save_checkpoint,
)
from spikepose_thesis.training.runner import _epoch, seed_worker
from spikepose_thesis.training.schedulers import ConstantScheduler


def test_seed_worker_limits_opencv_parallelism(monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, "setNumThreads", lambda value: calls.append(("threads", value)))
    monkeypatch.setattr(
        cv2.ocl, "setUseOpenCL", lambda value: calls.append(("opencl", value)),
    )
    monkeypatch.setattr(torch, "initial_seed", lambda: 123)

    seed_worker(0)

    assert calls == [("threads", 1), ("opencl", False)]


def test_checkpoint_restores_full_training_state(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = ConstantScheduler(optimizer, learning_rate=0.01)
    generator = torch.Generator().manual_seed(123)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    scheduler.set_epoch(10)
    expected_model = {key: value.detach().clone() for key, value in model.state_dict().items()}
    expected_generator = generator.get_state().clone()
    path = tmp_path / "last.pt"
    save_checkpoint(
        path, model, optimizer, scheduler, 10, 0.5,
        {"id": "example"}, [{"epoch": 10}],
        data_loader_generator=generator,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    generator.manual_seed(999)
    restored = restore_training_state(
        path, model, optimizer, scheduler, "cpu",
        data_loader_generator=generator,
    )

    assert restored["epoch"] == 10
    assert scheduler.last_epoch == 10
    assert torch.equal(generator.get_state(), expected_generator)
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected_model[key])
    assert optimizer.state


def test_amp_overflow_skips_step_and_backs_off_scale(monkeypatch):
    class FiniteForwardInfiniteBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, parameter):
            ctx.shape = parameter.shape
            return parameter.new_ones(())

        @staticmethod
        def backward(ctx, gradient):
            return gradient.new_full(ctx.shape, float("inf"))

    model = torch.nn.Linear(2, 1, bias=False)
    model.training_config = {}
    initial = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0)

    def overflow_loss(model, *_args, **_kwargs):
        return FiniteForwardInfiniteBackward.apply(model.weight)

    monkeypatch.setattr(
        "spikepose_thesis.training.runner._loss", overflow_loss,
    )
    loss, gradient_norm, _ = _epoch(
        model,
        [{"image": torch.ones(1, 1)}],
        criterion=None,
        device=torch.device("cpu"),
        optimizer=optimizer,
        gradient_clip=1.0,
        scaler=scaler,
        data={},
        progress_label="overflow-test",
    )

    assert loss == 1.0
    assert torch.isnan(torch.tensor(gradient_norm))
    assert model.last_amp_skipped_steps == 1
    assert model.last_amp_overflow_rows[0]["parameters"][0]["name"] == "weight"
    assert scaler.get_scale() == 4.0
    torch.testing.assert_close(model.weight, initial)
    assert model.weight.grad is None


def test_readiness_marks_short_completed_run_resumable(monkeypatch, tmp_path):
    config = {
        "id": "example", "paper_id": "Example", "dataset": "mpii",
        "stage": "spatial", "run_type": "pilot",
        "training": {"epochs": 60},
    }
    run = tmp_path / "runs" / "mpii" / "spatial" / "example" / "seed_42"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "last.pt").touch()
    (run / "training").mkdir()
    with (run / "training" / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerow({"epoch": 20})
    (run / "status.json").write_text(json.dumps({"state": "completed"}))
    report = {"checks": {
        "mpii_images": True, "mpii_train_metadata": True,
        "mpii_validation_metadata": True,
    }}

    readiness = experiment_readiness(config, 42, tmp_path, report)
    assert readiness == {
        "status": "RESUMABLE", "blockers": [],
        "completed_epoch": 20, "target_epoch": 60,
    }


def test_readiness_does_not_relaunch_a_running_checkpoint(tmp_path):
    config = {
        "id": "example", "paper_id": "Example", "dataset": "mpii",
        "stage": "spatial", "run_type": "pilot",
        "training": {"epochs": 20},
    }
    run = tmp_path / "runs" / "mpii" / "spatial" / "example" / "seed_42"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "last.pt").touch()
    (run / "training").mkdir()
    with (run / "training" / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerow({"epoch": 14})
    (run / "status.json").write_text(json.dumps({"state": "running"}))
    report = {"checks": {
        "mpii_images": True, "mpii_train_metadata": True,
        "mpii_validation_metadata": True,
    }}

    readiness = experiment_readiness(config, 42, tmp_path, report)
    assert readiness["status"] == "RUNNING"


def test_pilot_runtime_overrides_are_canonical_and_isolated(tmp_path):
    base = load_experiment("mamv2_fullcs_p00_source", profile="pilot20")
    scoped = apply_pilot_runtime_overrides(
        base, epochs=60, ntu_setups="S010-S015",
    )
    assert scoped["training"]["epochs"] == 60
    assert scoped["training"]["milestone_epochs"] == [10, 20, 60]
    assert scoped["data"]["setup_filter"] == [
        "S010", "S011", "S012", "S013", "S014", "S015",
    ]
    assert scoped["data"]["train_metadata"].endswith(
        "metadata/contiguous_xsub/train_split.csv"
    )
    assert scoped["data"]["validation_metadata"].endswith(
        "metadata/contiguous_xsub/val_split.csv"
    )
    assert scoped["data"]["pose_validation_mode"] == "manifest"
    assert scoped["data"]["exclude_overlapping_clips"] is True
    assert run_dir(scoped, 42, tmp_path) != run_dir(base, 42, tmp_path)
    assert "setups_S010_S011_S012_S013_S014_S015" in str(
        run_dir(scoped, 42, tmp_path)
    )
    shorter = apply_pilot_runtime_overrides(
        base, epochs=10, ntu_setups="S010-S015",
    )
    assert run_dir(scoped, 42, tmp_path) == run_dir(shorter, 42, tmp_path)

    run = run_dir(shorter, 42, tmp_path)
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "last.pt").touch()
    (run / "training").mkdir()
    with (run / "training" / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerow({"epoch": 10})
    (run / "status.json").write_text(json.dumps({"state": "completed"}))
    report = {"checks": {
        "ntu_skeletons_complete": True, "ntu_frames_complete": True,
        "ntu_clip_layout_valid": True, "ntu_clips_nonoverlapping": True,
        "ntu_quality_exclusions_complete": True,
        "ntu_pose_preflight_ready": True, "ntu_runtime_cache_ready": True,
        "ntu_cross_subject_disjoint": True,
        "ntu_cross_subject_metadata": True,
        "ntu_cross_subject_performance_groups_disjoint": True,
    }}
    readiness = experiment_readiness(scoped, 42, tmp_path, report)
    assert readiness["status"] == "RESUMABLE"
    assert readiness["completed_epoch"] == 10
    assert readiness["target_epoch"] == 60


def test_setup_parser_and_custom_epoch_node():
    assert parse_ntu_setups("full") is None
    assert parse_ntu_setups("S010-S012,S015") == (
        "S010", "S011", "S012", "S015",
    )
    config = apply_pilot_runtime_overrides(
        load_experiment("mamv2_fullcs_p00_source", profile="pilot20"),
        epochs=35, ntu_setups="S010",
    )
    assert config["training"]["milestone_epochs"] == [10, 20, 35]


def test_full_setup_override_expands_s010_metadata_to_canonical_cross_subject():
    base = load_experiment("mamv2_p06_noalign", profile="pilot20")
    assert "/metadata/s010/" in f"/{base['data']['train_metadata']}"

    full = apply_pilot_runtime_overrides(
        base, epochs=20, ntu_setups="full",
    )

    assert full["data"]["setup_filter"] is None
    assert full["data"]["train_metadata"].endswith(
        "metadata/contiguous_xsub/train_split.csv"
    )
    assert full["data"]["validation_metadata"].endswith(
        "metadata/contiguous_xsub/val_split.csv"
    )
    assert full["data"]["test_metadata"].endswith(
        "metadata/contiguous_xsub/test_split.csv"
    )
    assert full["data"]["exclusion_metadata"].endswith(
        "metadata/contiguous_xsub/excluded_samples.csv"
    )
    assert full["data"]["pose_validation_mode"] == "manifest"
    assert full["data"]["exclude_overlapping_clips"] is True
    assert full["run_variant"] == "setups_full"


def test_runtime_overrides_reject_formal_training():
    formal = load_experiment("confirm140_ntu_spikepose_frame")
    try:
        apply_pilot_runtime_overrides(formal, epochs=10, ntu_setups="full")
    except ValueError as error:
        assert "restricted to Pilot" in str(error)
    else:
        raise AssertionError("formal runtime override was accepted")
