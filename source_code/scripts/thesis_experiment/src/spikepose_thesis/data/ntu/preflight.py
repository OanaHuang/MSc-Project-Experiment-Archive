from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import PROJECT_ROOT, resolve_project_path
from spikepose_thesis.data.ntu.core import (
    extract_primary_pose_sequence,
    read_skeleton_file,
)


PREFLIGHT_SCHEMA_VERSION = 1


def metadata_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def metadata_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _metadata_rejection(row: dict[str, str]) -> str | None:
    skeleton_frames = _safe_int(row.get("skeleton_frames"))
    max_bodies = _safe_int(row.get("max_bodies"))
    empty_frames = _safe_int(row.get("empty_frames"))
    if skeleton_frames <= 0:
        return "skeleton_frames <= 0"
    if max_bodies <= 0:
        return "max_bodies <= 0"
    if empty_frames >= skeleton_frames:
        return "all skeleton frames are empty"
    return None


def validate_pose_file(skeleton_path: Path) -> tuple[bool, str]:
    """Apply the same strict pose checks used by the training Dataset."""
    if not skeleton_path.is_file():
        return False, f"skeleton file not found: {skeleton_path}"
    try:
        pose_sequence = extract_primary_pose_sequence(
            read_skeleton_file(skeleton_path),
        )
    except Exception as error:  # malformed source files must be reported verbatim
        return False, f"{type(error).__name__}: {error}"
    if "color_xy" not in pose_sequence:
        return False, "pose sequence has no color_xy"
    if "tracking_state" not in pose_sequence:
        return False, "pose sequence has no tracking_state"
    color_xy = np.asarray(pose_sequence["color_xy"])
    tracking_state = np.asarray(pose_sequence["tracking_state"])
    if color_xy.ndim != 3:
        return False, f"color_xy has invalid shape: {color_xy.shape}"
    if tracking_state.ndim != 2:
        return False, f"tracking_state has invalid shape: {tracking_state.shape}"
    if color_xy.shape[0] <= 0:
        return False, "pose sequence contains no frames"
    if tracking_state.shape[0] != color_xy.shape[0]:
        return False, "pose and tracking frame counts differ"
    if not np.isfinite(color_xy).any():
        return False, "all pose coordinates are non-finite"
    return True, ""


def _validate_record(record: tuple[str, str]) -> tuple[str, bool, str]:
    sample_id, source = record
    valid, reason = validate_pose_file(Path(source))
    return sample_id, valid, reason


def _protocol_data() -> tuple[dict[str, Any], ...]:
    """Return the canonical cross-subject data used by the paper roster."""
    return (load_experiment("confirm140_ntu_spikepose_frame")["data"],)


def _metadata_paths(protocols: Iterable[dict[str, Any]]) -> list[Path]:
    result: list[Path] = []
    for data in protocols:
        for split in ("train_metadata", "validation_metadata", "test_metadata"):
            path = resolve_project_path(data[split])
            if path not in result:
                result.append(path)
    return result


def default_manifest_path(data: dict[str, Any] | None = None) -> Path:
    data = data or _protocol_data()[0]
    configured = data.get(
        "pose_validation_manifest",
        "Datasets/NTU_RGBD/metadata/preflight/ntu_pose_validation_v1.json",
    )
    return resolve_project_path(configured)


def _manifest_matches(
    manifest: dict[str, Any], metadata_paths: Iterable[Path],
) -> bool:
    if manifest.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        return False
    if not manifest.get("ready", False):
        return False
    recorded = manifest.get("metadata", {})
    for path in metadata_paths:
        key = metadata_key(path)
        item = recorded.get(key)
        if not path.is_file() or not isinstance(item, dict):
            return False
        if item.get("sha256") != metadata_fingerprint(path):
            return False
    return True


def load_pose_preflight(
    data: dict[str, Any], metadata_paths: Iterable[Path] | None = None,
) -> dict[str, Any]:
    manifest_path = default_manifest_path(data)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "NTU pose preflight manifest is missing or invalid. Run "
            "`spikepose-thesis preflight-data` before training."
        ) from error
    paths = list(metadata_paths or _metadata_paths((data,)))
    if not _manifest_matches(manifest, paths):
        raise RuntimeError(
            "NTU pose preflight manifest does not match the current metadata. "
            "Run `spikepose-thesis preflight-data --force` before training."
        )
    return manifest


def load_validated_sample_ids(
    data: dict[str, Any], metadata_path: Path,
) -> frozenset[str]:
    manifest = load_pose_preflight(data, (metadata_path,))
    return frozenset(str(item) for item in manifest["valid_sample_ids"])


def cached_audit_report(ntu_data: dict[str, Any]) -> dict[str, Any] | None:
    paths = _metadata_paths((ntu_data,))
    try:
        manifest = load_pose_preflight(ntu_data, paths)
    except RuntimeError:
        return None
    report = manifest.get("audit_report")
    return report if isinstance(report, dict) else None


def preflight_available(ntu_data: dict[str, Any]) -> bool:
    return cached_audit_report(ntu_data) is not None


def _collect_records(
    paths: Iterable[Path],
) -> tuple[list[tuple[str, str]], list[dict[str, str]], dict[str, dict[str, Any]]]:
    sources: dict[str, str] = {}
    rejected: list[dict[str, str]] = []
    metadata: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = 0
        seen: set[str] = set()
        with path.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                sample_id = str(row.get("sample_id", "")).strip()
                if not sample_id:
                    rejected.append({
                        "sample_id": "<missing>",
                        "reason": f"missing sample_id in {metadata_key(path)}",
                    })
                    continue
                if sample_id in seen:
                    rejected.append({
                        "sample_id": sample_id,
                        "reason": f"duplicate sample_id in {metadata_key(path)}",
                    })
                    continue
                seen.add(sample_id)
                reason = _metadata_rejection(row)
                if reason:
                    rejected.append({"sample_id": sample_id, "reason": reason})
                    continue
                source = str(row.get("skeleton_path", "")).strip()
                if not source:
                    rejected.append({
                        "sample_id": sample_id, "reason": "missing skeleton_path",
                    })
                    continue
                previous = sources.setdefault(sample_id, source)
                if previous != source:
                    rejected.append({
                        "sample_id": sample_id,
                        "reason": "inconsistent skeleton_path across metadata splits",
                    })
        metadata[metadata_key(path)] = {
            "sha256": metadata_fingerprint(path),
            "rows": rows,
        }
    return sorted(sources.items()), rejected, metadata


def run_ntu_preflight(
    workers: int = 4,
    output_path: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    ntu_data = _protocol_data()[0]
    paths = _metadata_paths((ntu_data,))
    output = (output_path or default_manifest_path(ntu_data)).resolve()
    if output.is_file() and not force:
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
        if _manifest_matches(existing, paths):
            return {**existing, "reused": True}

    # Imported here to avoid a module cycle: audit_data consults this cache.
    from spikepose_thesis.data.audit import audit_data

    audit_report = audit_data(
        require_preflight=False,
        use_preflight_cache=False,
        require_runtime_cache=False,
    )
    if not audit_report["ready"]:
        raise RuntimeError(
            "The split/frame audit failed; refusing to approve pose data."
        )

    records, invalid, metadata = _collect_records(paths)
    valid_ids: list[str] = []
    print(
        f"Strictly validating {len(records)} unique NTU pose sequences "
        f"with {workers} workers...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_validate_record, records, chunksize=64)
        for index, (sample_id, valid, reason) in enumerate(results, 1):
            if valid:
                valid_ids.append(sample_id)
            else:
                invalid.append({"sample_id": sample_id, "reason": reason})
            if index % 1000 == 0 or index == len(records):
                print(f"Validated {index}/{len(records)}", flush=True)

    valid_ids.sort()
    invalid.sort(key=lambda item: (item["sample_id"], item["reason"]))
    manifest = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": bool(audit_report["ready"] and not invalid),
        "assumption": "skeleton and extracted-frame files are immutable",
        "protocols": ["ntu60_cs"],
        "metadata": metadata,
        "unique_samples": len(records),
        "valid_samples": len(valid_ids),
        "invalid_samples": invalid,
        "valid_sample_ids": valid_ids,
        "audit_report": audit_report,
        "workers": workers,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    return manifest
