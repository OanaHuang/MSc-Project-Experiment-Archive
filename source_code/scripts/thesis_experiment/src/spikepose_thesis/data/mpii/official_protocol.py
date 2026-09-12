"""Prepare the fixed MPII split published with the official HRNet code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


EXPECTED_ANNOTATION_SHA256 = {
    "train.json": "82e733b1eb684e2e82baec8dbf34e7b8b4df91500cf5a90ea68771ad14e2972e",
    "valid.json": "078061dfefba61e94c7ceff80ef95f7b7edd548f1f3de05089dff88619aaf3d8",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(item: dict) -> tuple[str, float, float]:
    center = item["center"]
    return item["image"], round(float(center[0]), 3), round(float(center[1]), 3)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare_official_mpii_protocol(
    official_root: Path, legacy_root: Path, output_root: Path, images_root: Path,
) -> dict:
    """Create ordered JSONL files without changing official split membership."""
    official_root = Path(official_root)
    legacy_root = Path(legacy_root)
    output_root = Path(output_root)
    images_root = Path(images_root)
    annotation_hashes = {
        name: _sha256(official_root / name) for name in EXPECTED_ANNOTATION_SHA256
    }
    mismatches = {
        name: {"expected": EXPECTED_ANNOTATION_SHA256[name], "actual": value}
        for name, value in annotation_hashes.items()
        if value != EXPECTED_ANNOTATION_SHA256[name]
    }
    if mismatches:
        raise RuntimeError(f"Official MPII annotation hash mismatch: {mismatches}")

    legacy = {}
    for name in ("train.jsonl", "val.jsonl"):
        for item in _read_jsonl(legacy_root / name):
            legacy[_signature(item)] = item

    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "official_hrnet_mpii",
        "annotation_sha256": annotation_hashes,
        "splits": {},
    }
    for source_name, target_name in (
        ("train.json", "train.jsonl"), ("valid.json", "val.jsonl"),
    ):
        official = json.loads(
            (official_root / source_name).read_text(encoding="utf-8")
        )
        output, unmatched, fallback_records, missing_images = [], [], [], []
        for official_index, row in enumerate(official):
            signature = _signature(row)
            match = legacy.get(signature)
            if match is None:
                unmatched.append(signature)
                if source_name == "valid.json":
                    continue
                match = {
                    "image": row["image"], "person_index": official_index,
                    "head_length": float("nan"),
                }
                fallback_records.append(signature)
            if not (images_root / row["image"]).is_file():
                missing_images.append(row["image"])
                continue
            output.append({
                **match,
                "center": [float(value) for value in row["center"]],
                "scale": float(row["scale"]),
                "keypoints": row["joints"],
                "visibility": row["joints_vis"],
                "official_index": official_index,
                "split_protocol": "official_hrnet_mpii",
            })
        with (output_root / target_name).open("w", encoding="utf-8") as handle:
            for row in output:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        report["splits"][target_name] = {
            "official_records": len(official),
            "prepared_records": len(output),
            "unmatched_records": len(unmatched),
            "fallback_records": len(fallback_records),
            "missing_images": sorted(set(missing_images)),
            "first_unmatched": unmatched[:10],
        }
    (output_root / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    return report
