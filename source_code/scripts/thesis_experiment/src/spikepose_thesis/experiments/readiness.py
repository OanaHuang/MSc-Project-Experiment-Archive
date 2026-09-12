from __future__ import annotations

import json
import csv
import os
from pathlib import Path

from spikepose_thesis.artifacts import locate_source_run, run_dir
from spikepose_thesis.core.paths import default_output_root
from spikepose_thesis.core.paths import resolve_project_path
from spikepose_thesis.data import audit_data


def _missing(report: dict, names: tuple[str, ...]) -> list[str]:
    checks = report["checks"]
    return [name for name in names if not checks.get(name, False)]


def _last_completed_epoch(run: Path, status: dict) -> int:
    if status.get("completed_epoch") is not None:
        return int(status["completed_epoch"])
    history_path = run / "training" / "history.csv"
    try:
        with history_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return int(rows[-1]["epoch"]) if rows else 0
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return 0


def _process_is_alive(pid: object) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def experiment_readiness(
    config: dict,
    seed: int | None = None,
    output_root: Path | None = None,
    report: dict | None = None,
) -> dict:
    """Return READY, WAITING or TBD without mutating experiment state."""
    implementation_status = config.get("implementation_status")
    if implementation_status not in {None, "ready"}:
        return {
            "status": "TBD",
            "blockers": [f"implementation_status:{implementation_status}"],
        }
    report = report or audit_data()
    if seed is not None:
        current_run = run_dir(config, seed, output_root)
        status_path = current_run / "status.json"
        try:
            current_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current_status = {}
        completed_epoch = _last_completed_epoch(current_run, current_status)
        target_epoch = int(config.get("training", {}).get("epochs", 0))
        last_checkpoint = current_run / "checkpoints" / "last.pt"
        if current_status.get("state") == "completed" and completed_epoch >= target_epoch:
            return {"status": "COMPLETE", "blockers": []}
        if current_status.get("state") in {"initialized", "running"}:
            # Old runs do not record a PID, so preserve their explicit running
            # state.  New runs record one and can safely fall back to RESUMABLE
            # when a hard-killed process leaves stale status behind.
            pid = current_status.get("pid")
            if pid is None or _process_is_alive(pid):
                return {
                    "status": "RUNNING",
                    "blockers": [f"existing_run:{status_path}"],
                }
        if last_checkpoint.is_file() and 0 < completed_epoch < target_epoch:
            return {
                "status": "RESUMABLE", "blockers": [],
                "completed_epoch": completed_epoch,
                "target_epoch": target_epoch,
            }
    dataset = str(config["dataset"])
    required = (
        ("mpii_images", "mpii_train_metadata", "mpii_validation_metadata")
        if dataset == "mpii"
        else (
            "ntu_skeletons_complete", "ntu_frames_complete",
            "ntu_clip_layout_valid", "ntu_clips_nonoverlapping",
            "ntu_quality_exclusions_complete",
            "ntu_pose_preflight_ready",
            "ntu_runtime_cache_ready",
            "ntu_cross_view_disjoint" if dataset == "ntu60_cv"
            else "ntu_cross_subject_disjoint",
            "ntu_cross_view_metadata" if dataset == "ntu60_cv"
            else "ntu_cross_subject_metadata",
            *(tuple() if dataset == "ntu60_cv" else (
                "ntu_cross_subject_performance_groups_disjoint",
            )),
        )
    )
    blockers = _missing(report, required)
    data = config.get("data", {})
    if dataset == "mpii" and data.get("split_protocol") == "official_hrnet_mpii":
        official_paths = {
            "official_mpii_train_metadata": data.get("train_metadata"),
            "official_mpii_validation_metadata": data.get("validation_metadata"),
        }
        blockers = [
            value for value in blockers
            if value not in {"mpii_train_metadata", "mpii_validation_metadata"}
        ]
        blockers.extend(
            f"{label}:{resolve_project_path(path)}"
            for label, path in official_paths.items()
            if not path or not resolve_project_path(path).is_file()
        )
    scoped_runtime_cache = (
        data.get("setup_filter") is not None
        or "/metadata/s010/" in f"/{str(data.get('train_metadata', ''))}"
    )
    if dataset != "mpii" and scoped_runtime_cache:
        from spikepose_thesis.data.ntu.runtime_cache import runtime_cache_status

        cache_status = runtime_cache_status(data)
        blockers = [
            value for value in blockers if value != "ntu_runtime_cache_ready"
        ]
        if not cache_status["ready"]:
            blockers.append(
                f"ntu_runtime_cache_ready:{cache_status['reason']}"
            )

    protocol = config.get("refinement_protocol", {})
    required_length = int(protocol.get("clip_length", 0))
    available_length = int(report.get("contiguous_clip_length", 0))
    trains_refiner = (
        config.get("action") == "refinement"
        and int(config.get("training", {}).get("epochs", 0)) > 0
    )
    if trains_refiner and required_length and available_length < required_length:
        blockers.append(
            f"contiguous_frames_{available_length}_lt_{required_length}"
        )

    if blockers:
        return {"status": "TBD", "blockers": blockers}

    priority = str(config.get("priority", ""))
    if priority == "conditional":
        return {"status": "TBD", "blockers": ["conditional_model_selection"]}

    initialization = config.get("initialization", {})
    if initialization.get("mode") == "official_checkpoint":
        checkpoint = resolve_project_path(initialization["path"])
        if not checkpoint.is_file():
            return {
                "status": "WAITING",
                "blockers": [f"official_checkpoint_missing:{checkpoint}"],
            }

    source = initialization.get("source")
    if source and seed is not None:
        effective_root = output_root or default_output_root(
            str(config.get("run_type", "formal")),
        )
        if initialization.get("source_output_root"):
            effective_root = resolve_project_path(initialization["source_output_root"])
        source_run = locate_source_run(source, seed, effective_root)
        status_path = source_run / "status.json"
        try:
            source_status = json.loads(status_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            source_status = {}
        allow_incomplete_source = bool(
            initialization.get("allow_incomplete_source", False)
        )
        if (
            source_status.get("state") != "completed"
            and not allow_incomplete_source
        ):
            return {
                "status": "WAITING",
                "blockers": [f"source_not_completed:{status_path}"],
            }
        source_run_type = source_status.get("run_type")
        expected_run_type = initialization.get(
            "source_run_type", config.get("run_type", "formal"),
        )
        if source_run_type is not None and source_run_type != expected_run_type:
            return {
                "status": "WAITING",
                "blockers": [
                    f"source_run_type_{source_run_type}_expected_{expected_run_type}"
                ],
            }
        load = config.get("initialization", {}).get("load", "weights_only")
        required_files = (
            [
                source_run / "predictions" / "train" / "predictions.npz",
                source_run / "predictions" / "validation" / "predictions.npz",
            ]
            if load == "predictions"
            else [source_run / "checkpoints" / "best.pt"]
        )
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            return {"status": "WAITING", "blockers": missing}

    return {"status": "READY", "blockers": []}
