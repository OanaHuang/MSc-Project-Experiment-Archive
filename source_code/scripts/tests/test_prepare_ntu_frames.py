from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.NTU_RGBD.prepare_setup_frames import (
    SAMPLE_MARKER, copy_sampled_frames, frame_number, sample_id_from_video,
)


class PrepareNTUFramesTests(unittest.TestCase):
    def test_video_name_and_frame_number_parsing(self) -> None:
        self.assertEqual(
            sample_id_from_video(Path("S010C001P007R001A001_rgb.avi")),
            "S010C001P007R001A001",
        )
        self.assertEqual(frame_number(Path("frame_000125.jpg")), 125)
        with self.assertRaises(ValueError):
            sample_id_from_video(Path("unexpected.avi"))
        with self.assertRaises(ValueError):
            frame_number(Path("frame_125.jpg"))

    def test_stride_copy_is_independent_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "full" / "S010C001P007R001A001"
            target = root / "stride5"
            source.mkdir(parents=True)
            for index in range(12):
                (source / f"frame_{index:06d}.jpg").write_bytes(
                    f"frame-{index}".encode(),
                )
            (source / SAMPLE_MARKER).write_text(json.dumps({
                "sample_id": source.name,
                "saved_frames": 12,
            }))

            result = copy_sampled_frames(source, target, 5)
            copied = sorted((target / source.name).glob("frame_*.jpg"))
            self.assertEqual([path.name for path in copied], [
                "frame_000000.jpg", "frame_000005.jpg", "frame_000010.jpg",
            ])
            self.assertEqual(result["saved_frames"], 3)
            self.assertNotEqual(copied[0].stat().st_ino,
                                (source / copied[0].name).stat().st_ino)
            self.assertEqual(
                copy_sampled_frames(source, target, 5)["status"], "skipped",
            )


if __name__ == "__main__":
    unittest.main()
