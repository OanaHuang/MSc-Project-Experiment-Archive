from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Iterable

import cv2
import numpy as np

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.config import parse_ntu_setups
from spikepose_thesis.core.paths import resolve_project_path
from spikepose_thesis.data.ntu.core import (
    coordinate_visibility,
    extract_primary_pose_sequence,
    read_skeleton_file,
)
from spikepose_thesis.data.ntu.datasets.person_crop import (
    compute_person_bbox,
    crop_and_resize_person_with_bbox,
)
from spikepose_thesis.data.ntu.preflight import (
    default_manifest_path,
    load_pose_preflight,
    metadata_fingerprint,
    metadata_key,
)


RUNTIME_CACHE_SCHEMA_VERSION = 2
SPATIAL_CROP_MODE = "clip_tube"


def default_runtime_cache_root(data: dict[str, Any] | None = None) -> Path:
    data = data or load_experiment("confirm140_ntu_spikepose_frame")["data"]
    configured = data.get(
        "runtime_cache_root", "Datasets/NTU_RGBD/runtime_cache_v1",
    )
    return resolve_project_path(configured)


def default_runtime_manifest_path(data: dict[str, Any] | None = None) -> Path:
    data = data or load_experiment("confirm140_ntu_spikepose_frame")["data"]
    configured = data.get("runtime_cache_manifest")
    if configured:
        return resolve_project_path(configured)
    return default_runtime_cache_root(data) / "manifest.json"


def _protocol_data() -> tuple[dict[str, Any], ...]:
    """Return every NTU protocol that must share the one physical cache."""
    return tuple(
        load_experiment(name)["data"]
        for name in (
            "confirm140_ntu_spikepose_frame",
            "confirm140_ntu_mamv2",
            "pilot8_ntu25_framewise",
        )
    )


def _metadata_paths(protocols: Iterable[dict[str, Any]]) -> list[Path]:
    result: list[Path] = []
    for data in protocols:
        for split in ("train_metadata", "validation_metadata", "test_metadata"):
            path = resolve_project_path(data[split])
            if path not in result:
                result.append(path)
    return result


def runtime_cache_status(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Perform a cheap integrity check without scanning source images."""
    data = data or _protocol_data()[0]
    manifest_path = default_runtime_manifest_path(data)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"ready": False, "reason": "manifest_missing", "path": str(manifest_path)}
    if manifest.get("schema_version") != RUNTIME_CACHE_SCHEMA_VERSION:
        return {"ready": False, "reason": "schema_mismatch", "path": str(manifest_path)}
    if not manifest.get("ready", False):
        return {"ready": False, "reason": "manifest_incomplete", "path": str(manifest_path)}

    required_setups: set[str] = set()
    for path in _metadata_paths((data,)):
        item = manifest.get("metadata", {}).get(metadata_key(path), {})
        if not path.is_file() or item.get("sha256") != metadata_fingerprint(path):
            return {"ready": False, "reason": "metadata_changed", "path": str(path)}
        required_setups.update(str(value) for value in item.get("setups", []))
    configured_setups = data.get("setup_filter")
    if configured_setups is not None:
        required_setups.intersection_update(str(value) for value in configured_setups)
    available_setups = {
        str(value) for value in manifest.get("available_setups", [])
    }
    missing_setups = sorted(required_setups - available_setups)
    if missing_setups:
        return {
            "ready": False,
            "reason": "setup_scope_missing",
            "path": str(manifest_path),
            "missing_setups": missing_setups,
        }
    for key, expected in (
        ("image_size", int(data["image_size"])),
        ("bbox_expansion", float(data["bbox_expansion"])),
        ("frame_clip_subdir", str(data["frame_clip_subdir"])),
        ("crop_mode", SPATIAL_CROP_MODE),
    ):
        if manifest.get(key) != expected:
            return {
                "ready": False, "reason": f"{key}_mismatch",
                "path": str(manifest_path),
            }
    preflight_path = default_manifest_path(data)
    if (
        not preflight_path.is_file()
        or manifest.get("pose_preflight_sha256") != metadata_fingerprint(preflight_path)
    ):
        return {"ready": False, "reason": "preflight_changed", "path": str(preflight_path)}
    root = default_runtime_cache_root(data)
    for item in manifest.get("databases", []):
        path = root / str(item.get("file", ""))
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            return {"ready": False, "reason": "database_missing", "path": str(path)}
    return {"ready": True, "reason": "", "path": str(manifest_path), "manifest": manifest}


def require_runtime_cache(data: dict[str, Any]) -> dict[str, Any]:
    status = runtime_cache_status(data)
    if not status["ready"]:
        raise RuntimeError(
            "NTU runtime cache is missing or stale "
            f"({status['reason']}: {status['path']}). Run "
            "`spikepose-thesis prepare-runtime-cache` before training."
        )
    return status["manifest"]


def _array_blob(value: np.ndarray, dtype: str) -> bytes:
    return np.ascontiguousarray(value, dtype=np.dtype(dtype)).tobytes()


class RuntimeCacheReader:
    """Worker-local read-only access to setup-sharded SQLite databases."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._connections: dict[str, sqlite3.Connection] = {}
        self._crop_modes: dict[str, str] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_connections"] = {}
        return state

    def _connection(self, sample_id: str) -> sqlite3.Connection:
        setup = sample_id[:4]
        if setup not in self._connections:
            path = (self.root / f"{setup}.sqlite3").resolve()
            uri = path.as_uri() + "?mode=ro&immutable=1"
            self._connections[setup] = sqlite3.connect(uri, uri=True)
        return self._connections[setup]

    def load_pose(self, sample_id: str) -> dict[str, Any] | None:
        row = self._connection(sample_id).execute(
            "SELECT body_id, num_frames, num_joints, color_xy, tracking_state "
            "FROM pose WHERE sample_id = ?", (sample_id,),
        ).fetchone()
        if row is None:
            return None
        body_id, frames, joints, color_blob, tracking_blob = row
        return {
            "body_id": str(body_id),
            "color_xy": np.frombuffer(color_blob, dtype="<f4").reshape(
                int(frames), int(joints), 2,
            ).copy(),
            "tracking_state": np.frombuffer(tracking_blob, dtype="i1").reshape(
                int(frames), int(joints),
            ).copy(),
        }

    def spatial_crop_mode(self, sample_id: str) -> str:
        setup = sample_id[:4]
        if setup not in self._crop_modes:
            row = self._connection(sample_id).execute(
                "SELECT value FROM cache_metadata WHERE key='crop_mode'",
            ).fetchone()
            if row is None:
                raise RuntimeError(f"Runtime cache has no crop mode: {setup}")
            self._crop_modes[setup] = str(row[0])
        return self._crop_modes[setup]

    def load_spatial_frame(
        self, sample_id: str, clip_name: str | None, frame_number: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        row = self._connection(sample_id).execute(
            "SELECT image_webp, bbox FROM spatial_frame "
            "WHERE sample_id = ? AND clip_name = ? AND frame_number = ?",
            (sample_id, clip_name or "", int(frame_number)),
        ).fetchone()
        if row is None:
            return None
        encoded = np.frombuffer(row[0], dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode cached crop: {sample_id} frame={frame_number}")
        bbox = np.frombuffer(row[1], dtype="<f4").copy()
        return image, bbox

    def load_spatial_bbox(
        self, sample_id: str, clip_name: str | None, frame_number: int,
    ) -> np.ndarray | None:
        """Load cached crop geometry without decoding the cached image."""
        row = self._connection(sample_id).execute(
            "SELECT bbox FROM spatial_frame "
            "WHERE sample_id = ? AND clip_name = ? AND frame_number = ?",
            (sample_id, clip_name or "", int(frame_number)),
        ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype="<f4").copy()


def transform_keypoints_to_bbox(
    keypoints: np.ndarray,
    visibility: np.ndarray,
    bbox_xyxy: np.ndarray,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the exact keypoint transform used by cached person crops."""
    points = np.asarray(keypoints, dtype=np.float32).copy()
    visible = np.asarray(visibility, dtype=np.float32).copy()
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float32)
    width = float(x2 - x1)
    height = float(y2 - y1)
    if width <= 0 or height <= 0:
        raise ValueError("cached bbox has non-positive dimensions")
    points[:, 0] = (points[:, 0] - x1) * (float(output_size) / width)
    points[:, 1] = (points[:, 1] - y1) * (float(output_size) / height)
    inside = (
        np.isfinite(points).all(axis=1)
        & (points[:, 0] >= 0)
        & (points[:, 0] < output_size)
        & (points[:, 1] >= 0)
        & (points[:, 1] < output_size)
    )
    return points, (visible.astype(bool) & inside).astype(np.float32)


def _valid_database(path: Path, expected_signature: str) -> dict[str, int] | None:
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(path)
        complete = connection.execute(
            "SELECT value FROM cache_metadata WHERE key='complete'",
        ).fetchone()
        if complete != ("1",):
            return None
        signature = connection.execute(
            "SELECT value FROM cache_metadata WHERE key='signature'",
        ).fetchone()
        if signature != (expected_signature,):
            return None
        poses = int(connection.execute("SELECT COUNT(*) FROM pose").fetchone()[0])
        frames = int(connection.execute("SELECT COUNT(*) FROM spatial_frame").fetchone()[0])
        return {"poses": poses, "frames": frames}
    except sqlite3.DatabaseError:
        return None
    finally:
        if "connection" in locals():
            connection.close()


def _encode_tube_frame(
    item: tuple[Path, int, np.ndarray, np.ndarray, np.ndarray, int],
) -> tuple[int, bytes, bytes]:
    frame_path, frame_number, points, visible, tube_bbox, image_size = item
    image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read extracted frame: {frame_path}")
    crop = crop_and_resize_person_with_bbox(
        image, points, visible, tube_bbox, output_size=image_size,
    )
    ok, encoded = cv2.imencode(
        ".webp", crop.image, [int(cv2.IMWRITE_WEBP_QUALITY), 101],
    )
    if not ok:
        raise RuntimeError(f"Could not encode cached crop: {frame_path}")
    return (
        frame_number, encoded.tobytes(), _array_blob(crop.bbox_xyxy, "<f4"),
    )


def _build_setup(payload: dict[str, Any]) -> dict[str, Any]:
    setup = str(payload["setup"])
    target = Path(payload["target"])
    if not payload["force"]:
        counts = _valid_database(target, str(payload["signature"]))
        if counts is not None:
            return {"setup": setup, "file": target.name, **counts, "reused": True}

    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            "PRAGMA journal_mode=OFF;"
            "PRAGMA synchronous=OFF;"
            "PRAGMA temp_store=MEMORY;"
            "PRAGMA page_size=65536;"
            "CREATE TABLE cache_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE pose ("
            "sample_id TEXT PRIMARY KEY, body_id TEXT NOT NULL, "
            "num_frames INTEGER NOT NULL, num_joints INTEGER NOT NULL, "
            "color_xy BLOB NOT NULL, tracking_state BLOB NOT NULL"
            ") WITHOUT ROWID;"
            "CREATE TABLE spatial_frame ("
            "sample_id TEXT NOT NULL, clip_name TEXT NOT NULL, frame_number INTEGER NOT NULL, "
            "image_webp BLOB NOT NULL, bbox BLOB NOT NULL, "
            "PRIMARY KEY (sample_id, clip_name, frame_number)"
            ") WITHOUT ROWID;"
        )
        pose_count = frame_count = 0
        with ThreadPoolExecutor(
            max_workers=int(payload.get("frame_workers", 1)),
        ) as frame_executor:
            for record in payload["records"]:
                sample_id = str(record["sample_id"])
                pose = extract_primary_pose_sequence(
                    read_skeleton_file(Path(record["skeleton_path"])),
                )
                color_xy = np.asarray(pose["color_xy"], dtype="<f4")
                tracking = np.asarray(pose["tracking_state"], dtype="i1")
                connection.execute(
                    "INSERT INTO pose VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        sample_id, str(pose.get("body_id", "primary")),
                        len(color_xy), color_xy.shape[1],
                        _array_blob(color_xy, "<f4"), _array_blob(tracking, "i1"),
                    ),
                )
                pose_count += 1
                width = int(record["width"])
                height = int(record["height"])
                sample_root = (
                    Path(payload["frames_root"]) / setup
                    / payload["clip_subdir"] / sample_id
                )
                for clip_dir in sorted(
                    item for item in sample_root.glob("clip_*") if item.is_dir()
                ):
                    frame_paths = sorted(clip_dir.glob("frame_*.jpg"))
                    if len(frame_paths) != int(payload["clip_length"]):
                        raise RuntimeError(
                            f"Expected {payload['clip_length']} frames in {clip_dir}; "
                            f"found {len(frame_paths)}"
                        )
                    frame_numbers = [
                        int(frame_path.stem.rsplit("_", 1)[1])
                        for frame_path in frame_paths
                    ]
                    tube_points: list[np.ndarray] = []
                    tube_visibility: list[np.ndarray] = []
                    for frame_number in frame_numbers:
                        if frame_number >= len(color_xy):
                            raise IndexError(
                                f"Pose frame out of range: {sample_id} {frame_number}"
                            )
                        points = color_xy[frame_number]
                        visible = coordinate_visibility(
                            points, tracking_state=tracking[frame_number],
                            image_size=(width, height), include_inferred=False,
                        ).astype(np.float32)
                        tube_points.append(points)
                        tube_visibility.append(visible)
                    tube_bbox = compute_person_bbox(
                        np.concatenate(tube_points, axis=0),
                        np.concatenate(tube_visibility, axis=0),
                        width, height, float(payload["bbox_expansion"]), True,
                    )
                    jobs = [
                        (
                            frame_path, frame_number, points, visible, tube_bbox,
                            int(payload["image_size"]),
                        )
                        for frame_path, frame_number, points, visible in zip(
                            frame_paths, frame_numbers, tube_points, tube_visibility,
                        )
                    ]
                    for frame_number, encoded, bbox in frame_executor.map(
                        _encode_tube_frame, jobs,
                    ):
                        connection.execute(
                            "INSERT INTO spatial_frame VALUES (?, ?, ?, ?, ?)",
                            (sample_id, clip_dir.name, frame_number, encoded, bbox),
                        )
                        frame_count += 1
                connection.commit()
        connection.executemany(
            "INSERT INTO cache_metadata VALUES (?, ?)",
            (
                ("complete", "1"),
                ("schema_version", str(RUNTIME_CACHE_SCHEMA_VERSION)),
                ("image_size", str(payload["image_size"])),
                ("bbox_expansion", str(payload["bbox_expansion"])),
                ("crop_mode", SPATIAL_CROP_MODE),
                ("signature", str(payload["signature"])),
            ),
        )
        connection.commit()
    except Exception:
        connection.close()
        if temporary.exists():
            temporary.unlink()
        raise
    else:
        connection.close()
        temporary.replace(target)
    return {
        "setup": setup, "file": target.name,
        "poses": pose_count, "frames": frame_count, "reused": False,
    }


def _collect_records(paths: Iterable[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {}
    for path in paths:
        rows = 0
        setups: set[str] = set()
        with path.open(encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                rows += 1
                sample_id = str(row["sample_id"]).strip()
                setups.add(sample_id[:4])
                record = {
                    "sample_id": sample_id,
                    "skeleton_path": str(resolve_project_path(row["skeleton_path"])),
                    "width": int(float(row["width"])),
                    "height": int(float(row["height"])),
                }
                previous = records.setdefault(sample_id, record)
                if previous != record:
                    raise RuntimeError(f"Inconsistent metadata for {sample_id}")
        metadata[metadata_key(path)] = {
            "sha256": metadata_fingerprint(path), "rows": rows,
            "setups": sorted(setups),
        }
    return records, metadata


def prepare_runtime_cache(
    workers: int = 4, force: bool = False, setups: str | None = None,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    protocols = _protocol_data()
    ntu_data = protocols[0]
    paths = _metadata_paths(protocols)
    # MAM's S010 metadata is a subset protocol with validation intentionally
    # skipped. The strict formal manifests remain the cache trust anchor.
    load_pose_preflight(ntu_data, _metadata_paths(protocols[:2]))
    records, metadata = _collect_records(paths)
    selected = parse_ntu_setups(setups or "full")
    if selected is not None:
        selected_set = frozenset(selected)
        records = {
            sample_id: record for sample_id, record in records.items()
            if sample_id[:4] in selected_set
        }
    if not records:
        raise RuntimeError("No NTU records matched the requested setup scope")
    root = default_runtime_cache_root(ntu_data)
    root.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records.values():
        grouped.setdefault(record["sample_id"][:4], []).append(record)
    payloads = []
    for setup, items in sorted(grouped.items()):
        sorted_items = sorted(items, key=lambda item: item["sample_id"])
        signature = hashlib.sha256(json.dumps({
            "schema_version": RUNTIME_CACHE_SCHEMA_VERSION,
            "records": sorted_items,
            "image_size": int(ntu_data["image_size"]),
            "bbox_expansion": float(ntu_data["bbox_expansion"]),
            "clip_subdir": str(ntu_data["frame_clip_subdir"]),
            "clip_length": int(ntu_data["contiguous_clip_length"]),
            "crop_mode": SPATIAL_CROP_MODE,
        }, sort_keys=True).encode("utf-8")).hexdigest()
        payloads.append({
            "setup": setup,
            "target": str(root / f"{setup}.sqlite3"),
            "records": sorted_items,
            "frames_root": str(resolve_project_path(ntu_data["frames_dir"])),
            "clip_subdir": str(ntu_data["frame_clip_subdir"]),
            "image_size": int(ntu_data["image_size"]),
            "bbox_expansion": float(ntu_data["bbox_expansion"]),
            "clip_length": int(ntu_data["contiguous_clip_length"]),
            "force": bool(force),
            "signature": signature,
        })
    setup_workers = min(workers, len(payloads))
    frame_workers = max(1, workers // setup_workers)
    for payload in payloads:
        payload["frame_workers"] = frame_workers
    print(
        f"Preparing lossless NTU tube-crop cache for {len(records)} sequences "
        f"across {len(payloads)} setup shards with {setup_workers} setup workers "
        f"and {frame_workers} frame workers each...",
        flush=True,
    )
    databases: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=setup_workers) as executor:
        futures = {executor.submit(_build_setup, payload): payload["setup"] for payload in payloads}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            path = root / result["file"]
            result["bytes"] = path.stat().st_size
            databases.append(result)
            print(
                f"Runtime cache setup {result['setup']} complete "
                f"({index}/{len(payloads)}, poses={result['poses']}, "
                f"frames={result['frames']}, reused={result['reused']})",
                flush=True,
            )
    databases.sort(key=lambda item: item["setup"])
    preflight_path = default_manifest_path(ntu_data)
    manifest = {
        "schema_version": RUNTIME_CACHE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": True,
        "format": "setup-sharded SQLite; float32 pose; lossless WebP tube crops",
        "crop_mode": SPATIAL_CROP_MODE,
        "leakage_policy": (
            "Each sample is transformed independently with fixed geometry only; "
            "split membership is never read by another split's Dataset."
        ),
        "pose_preflight_sha256": metadata_fingerprint(preflight_path),
        "metadata": metadata,
        "unique_samples": len(records),
        "image_size": int(ntu_data["image_size"]),
        "bbox_expansion": float(ntu_data["bbox_expansion"]),
        "frame_clip_subdir": str(ntu_data["frame_clip_subdir"]),
        "clip_length": int(ntu_data["contiguous_clip_length"]),
        "available_setups": sorted(grouped),
        "databases": databases,
        "poses": sum(int(item["poses"]) for item in databases),
        "spatial_frames": sum(int(item["frames"]) for item in databases),
        "bytes": sum(int(item["bytes"]) for item in databases),
        "workers": workers,
    }
    output = default_runtime_manifest_path(ntu_data)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output)
    return manifest


__all__ = [
    "RuntimeCacheReader", "default_runtime_cache_root", "prepare_runtime_cache",
    "require_runtime_cache", "runtime_cache_status", "transform_keypoints_to_bbox",
]
