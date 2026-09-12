from pathlib import Path

import numpy as np

from scripts.NTU_RGBD.datasets.ntu_frame_dataset import NTUFrameDataset
from scripts.NTU_RGBD.prepare_contiguous_metadata import grouped_protocol_split
from scripts.NTU_RGBD.sample_contiguous_frames import clip_starts


def test_uniform_clips_are_contiguous_and_non_overlapping_when_possible():
    starts = clip_starts(frame_count=100, clip_length=16, clips=2)
    assert starts == [17, 67]
    assert starts[0] + 16 <= starts[1]


def test_contiguous_layout_keeps_clip_membership(tmp_path: Path):
    sample_id = "S001C001P001R001A001"
    sample_dir = tmp_path / "S001" / "contiguous_2x16" / sample_id
    for clip, frames in (("clip_00", range(10, 14)), ("clip_01", range(30, 34))):
        directory = sample_dir / clip
        directory.mkdir(parents=True)
        for frame in frames:
            (directory / f"frame_{frame:06d}.jpg").touch()

    dataset = object.__new__(NTUFrameDataset)
    dataset.extracted_frames_dir = tmp_path
    dataset.contiguous_clip_subdir = "contiguous_2x16"
    dataset.temporal_frame_gap = 1
    dataset._frame_locations = {}
    locations = dataset._sample_frame_locations(sample_id, sample_dir)
    dataset._frame_locations.update({(sample_id, frame): value for frame, value in locations.items()})

    assert dataset._history_stays_in_one_clip(sample_id, target=13, steps=4)
    assert not dataset._history_stays_in_one_clip(sample_id, target=30, steps=2)
    assert dataset._get_frame_path(sample_id, 32).parent.name == "clip_01"


def test_cross_subject_split_keeps_test_and_validation_subject_disjoint():
    rows = [
        {"sample_id": f"sample-{performer}", "performer": performer, "camera": 1}
        for performer in (1, 2, 3, 4, 6, 8, 9, 10)
    ]
    split = grouped_protocol_split(rows, "xsub", validation_fraction=0.25, seed=7)
    train_subjects = {row["performer"] for row in split["train"]}
    validation_subjects = {row["performer"] for row in split["validation"]}
    test_subjects = {row["performer"] for row in split["test"]}

    assert train_subjects.isdisjoint(validation_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert validation_subjects.isdisjoint(test_subjects)
    assert test_subjects == {3, 6, 10}
