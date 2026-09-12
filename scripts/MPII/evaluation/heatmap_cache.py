from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class HeatmapCacheManifest:
    schema_version: int
    samples: int
    num_joints: int
    heatmap_height: int
    heatmap_width: int
    dtype: str
    has_flipped_input: bool
    checkpoint_sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _write_targets(cache_dir: Path, values: dict[str, list]) -> None:
    arrays = {
        key: np.concatenate(items, axis=0)
        for key, items in values.items()
    }
    np.savez_compressed(cache_dir / "targets.npz", **arrays)


@torch.no_grad()
def generate_heatmap_cache(model, loader, device, cache_dir: Path,
                           checkpoint: Path, dtype="float16",
                           include_flipped=True) -> HeatmapCacheManifest:
    if dtype not in {"float16", "float32"}:
        raise ValueError("cache dtype must be float16 or float32")
    cache_dir.mkdir(parents=True, exist_ok=False)
    model.eval()
    original = flipped = None
    offset = 0
    targets = {key: [] for key in (
        "keypoints_original", "visibility", "head_length", "inverse",
    )}
    numpy_dtype = np.float16 if dtype == "float16" else np.float32
    for batch in loader:
        images = batch["image"].to(device)
        output = model(images).detach().cpu().numpy()
        if original is None:
            total = len(loader.dataset)
            shape = (total, *output.shape[1:])
            original = np.lib.format.open_memmap(
                cache_dir / "original_heatmaps.npy", mode="w+",
                dtype=numpy_dtype, shape=shape,
            )
            if include_flipped:
                flipped = np.lib.format.open_memmap(
                    cache_dir / "flipped_input_heatmaps.npy", mode="w+",
                    dtype=numpy_dtype, shape=shape,
                )
        count = len(output)
        original[offset:offset + count] = output.astype(numpy_dtype)
        if include_flipped:
            flipped_output = model(torch.flip(images, dims=(-1,)))
            flipped[offset:offset + count] = (
                flipped_output.detach().cpu().numpy().astype(numpy_dtype)
            )
        for key in targets:
            value = batch[key]
            targets[key].append(
                value.detach().cpu().numpy() if torch.is_tensor(value)
                else np.asarray(value)
            )
        offset += count
    if original is None or offset != len(loader.dataset):
        raise RuntimeError("Heatmap cache did not receive the complete dataset")
    original.flush()
    if flipped is not None:
        flipped.flush()
    _write_targets(cache_dir, targets)
    manifest = HeatmapCacheManifest(
        schema_version=1, samples=offset, num_joints=original.shape[1],
        heatmap_height=original.shape[2], heatmap_width=original.shape[3],
        dtype=dtype, has_flipped_input=bool(include_flipped),
        checkpoint_sha256=sha256_file(checkpoint),
    )
    (cache_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8",
    )
    return manifest


def load_heatmap_cache(cache_dir: Path, checkpoint: Path | None = None,
                       expected_samples: int | None = None) -> dict:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing heatmap cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported heatmap cache schema")
    if checkpoint is not None:
        actual = sha256_file(checkpoint)
        if actual != manifest.get("checkpoint_sha256"):
            raise ValueError("Heatmap cache checkpoint hash does not match")
    original = np.load(cache_dir / "original_heatmaps.npy", mmap_mode="r")
    flipped_path = cache_dir / "flipped_input_heatmaps.npy"
    flipped = np.load(flipped_path, mmap_mode="r") if flipped_path.is_file() else None
    targets_file = np.load(cache_dir / "targets.npz")
    targets = {key: targets_file[key] for key in targets_file.files}
    if len(original) != int(manifest["samples"]):
        raise ValueError("Heatmap cache sample count does not match manifest")
    if expected_samples is not None and len(original) != expected_samples:
        raise ValueError(
            f"Heatmap cache has {len(original)} samples, expected {expected_samples}"
        )
    return {"manifest": manifest, "original": original,
            "flipped_input": flipped, "targets": targets}
