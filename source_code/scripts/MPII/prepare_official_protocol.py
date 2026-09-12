"""Prepare fixed HRNet MPII train/valid JSONL from official annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _signature(item):
    center = item["center"]
    return item["image"], round(float(center[0]), 3), round(float(center[1]), 3)


def _read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prepare(official_root: Path, legacy_root: Path, output_root: Path,
            images_root: Path) -> dict:
    legacy = {}
    for name in ("train.jsonl", "val.jsonl"):
        for item in _read_jsonl(legacy_root / name):
            legacy[_signature(item)] = item
    output_root.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "official_hrnet_mpii", "splits": {}}
    for source_name, target_name in (("train.json", "train.jsonl"),
                                     ("valid.json", "val.jsonl")):
        official = json.loads((official_root / source_name).read_text(encoding="utf-8"))
        output, unmatched, fallback_records, missing_images = [], [], [], []
        for official_index, row in enumerate(official):
            match = legacy.get(_signature(row))
            if match is None:
                unmatched.append(_signature(row))
                if source_name == "valid.json":
                    continue
                # Official training does not need a head scale. Retain unusual
                # people omitted by the legacy metadata conversion instead of
                # silently changing the official training membership.
                match = {
                    "image": row["image"], "person_index": official_index,
                    "head_length": float("nan"),
                }
                fallback_records.append(_signature(row))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=PROJECT_ROOT /
                        "Datasets/MPII/official_simplebaseline/annot")
    parser.add_argument("--legacy-root", type=Path, default=PROJECT_ROOT /
                        "Datasets/MPII/metadata")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT /
                        "Datasets/MPII/metadata/official_hrnet")
    parser.add_argument("--images-root", type=Path, default=PROJECT_ROOT /
                        "Datasets/MPII/images")
    args = parser.parse_args()
    report = prepare(args.official_root, args.legacy_root, args.output_root,
                     args.images_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
