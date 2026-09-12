#!/usr/bin/env python3
"""Build a verified manifest for locally transcoded visualization videos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    source = json.loads((model_dir / "render_manifest.json").read_text())
    videos = []
    for item in source["videos"]:
        path = model_dir / "videos" / Path(item["path"]).name
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ]))["streams"][0]
        numerator, denominator = map(int, probe["avg_frame_rate"].split("/"))
        videos.append({
            "sample_id": item["sample_id"], "setup": item["setup"],
            "path": str(path), "codec": probe["codec_name"],
            "frames": int(probe["nb_read_frames"]),
            "width": int(probe["width"]), "height": int(probe["height"]),
            "fps": numerator / denominator, "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    payload = {
        "source_experiment_id": source["experiment_id"],
        "source_checkpoint": source["checkpoint"],
        "source_checkpoint_epoch": source["checkpoint_epoch"],
        "videos": videos,
    }
    (model_dir / "local_export_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
