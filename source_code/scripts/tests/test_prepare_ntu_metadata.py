from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.NTU_RGBD.prepare_setup_metadata import (
    build_rows,
    filter_usable_rows,
)


def write_frame_marker(root: Path, sample_id: str, frames: int = 1) -> Path:
    sample_dir = root / sample_id
    sample_dir.mkdir(parents=True)
    (sample_dir / ".frames_complete.json").write_text(json.dumps({
        "sample_id": sample_id,
        "saved_frames": frames,
        "source_frames": frames,
    }))
    return sample_dir


def write_skeleton(path: Path, frames: int = 1, bodies: int = 1) -> None:
    lines = [str(frames)]
    for _ in range(frames):
        lines.append(str(bodies))
        for body in range(bodies):
            lines.extend((f"body-{body}", "25"))
            lines.extend("0 0 0 0 0 0 0 1 0 0 0 2" for _ in range(25))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PrepareNTUMetadataTests(unittest.TestCase):
    def test_missing_invalid_and_empty_skeletons_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames"
            videos = root / "videos"
            skeletons = root / "skeletons"
            videos.mkdir()
            skeletons.mkdir()
            valid = "S010C001P001R001A001"
            missing = "S010C001P001R001A002"
            invalid = "S010C001P001R001A003"
            empty = "S010C001P001R001A004"
            for sample_id in (valid, missing, invalid, empty):
                write_frame_marker(frames, sample_id)
            write_skeleton(skeletons / f"{valid}.skeleton")
            (skeletons / f"{invalid}.skeleton").write_text("broken\n")
            write_skeleton(skeletons / f"{empty}.skeleton", frames=1, bodies=0)

            rows, exclusions = build_rows(
                frames, videos, skeletons, "S010", tolerance=2,
            )

            self.assertEqual([row["sample_id"] for row in rows], [valid])
            self.assertEqual(
                {item["sample_id"]: item["reason"] for item in exclusions},
                {
                    missing: "missing_skeleton",
                    invalid: "invalid_skeleton",
                    empty: "empty_skeleton",
                },
            )

    def test_quality_filter_excludes_multi_body_and_frame_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {
                    "sample_id": "S010C001P001R001A001",
                    "is_single_person": True,
                    "frame_count_match": True,
                    "rgb_frames": 10,
                    "skeleton_frames": 10,
                    "max_bodies": 1,
                    "skeleton_path": str(root / "valid.skeleton"),
                },
                {
                    "sample_id": "S010C001P001R001A002",
                    "is_single_person": False,
                    "frame_count_match": False,
                    "rgb_frames": 10,
                    "skeleton_frames": 20,
                    "max_bodies": 2,
                    "skeleton_path": str(root / "invalid.skeleton"),
                },
            ]

            usable, exclusions = filter_usable_rows(rows, root)

            self.assertEqual([row["sample_id"] for row in usable], [rows[0]["sample_id"]])
            self.assertEqual(len(exclusions), 1)
            self.assertEqual(
                exclusions[0]["reason"],
                "multiple_bodies;frame_count_mismatch",
            )


if __name__ == "__main__":
    unittest.main()
