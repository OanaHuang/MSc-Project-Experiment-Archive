from spikepose_thesis.data import audit_data
from spikepose_thesis.data.audit import _clip_layout
from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.core.paths import default_output_root
from spikepose_thesis.data.ntu.datasets.ntu_frame_dataset import NTUFrameDataset
from spikepose_thesis.experiments import plan_study, validate_repository


def test_paper_matrix_and_isolation():
    report = validate_repository(forward=False)
    assert report["experiments"] >= report["required_experiments"]
    assert report["legacy_imports"] == 0
    assert report["joint_layout_target"] == "ntu18_pending_mapping"
    assert report["retained_auxiliary_layout"] == "ntu25_identity_v1"


def test_canonical_studies_are_dependency_ordered():
    expected_sizes = {
        "icassp2027_pilot20": 16,
        "icassp2027_confirm140": 16,
        "icassp2027_pilot_eval": 7,
        "icassp2027_confirm_eval": 6,
        "ntu25_preview8": 2,
    }
    for study, expected_size in expected_sizes.items():
        configs = plan_study(study)
        assert len(configs) == expected_size
        position = {config["id"]: index for index, config in enumerate(configs)}
        for config in configs:
            source = config.get("initialization", {}).get("source")
            if source in position:
                assert position[source] < position[config["id"]]


def test_data_audit_has_formal_gate_schema():
    report = audit_data()
    assert isinstance(report["ready"], bool)
    assert report["expected_ntu_sequences"] == 56880
    assert report["expected_quality_exclusions"] == 11960
    assert report["expected_usable_ntu_sequences"] == 44920
    assert {
        "ntu_skeletons_complete",
        "ntu_frames_complete",
        "ntu_clip_layout_valid",
        "ntu_clips_nonoverlapping",
        "ntu_quality_exclusions_complete",
        "ntu_pose_preflight_ready",
        "ntu_runtime_cache_ready",
        "ntu_cross_subject_metadata",
        "ntu_cross_subject_disjoint",
        "ntu_cross_subject_performance_groups_disjoint",
    } <= report["checks"].keys()


def test_pilot_and_formal_output_roots_are_isolated(monkeypatch):
    monkeypatch.delenv("SPIKEPOSE_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("SPIKEPOSE_PILOT_OUTPUT_ROOT", raising=False)
    assert default_output_root("pilot").name == "Outputs_Thesis_Pilot20"
    assert default_output_root("formal").name == "Outputs_Thesis"


def test_ntu_training_requires_one_time_pose_manifest():
    for name in (
        "confirm140_ntu_spikepose_frame",
        "confirm140_ntu_mamv2",
        "pilot8_ntu25_framewise",
    ):
        data = load_experiment(name)["data"]
        assert data["pose_validation_mode"] == "manifest"
        assert data["pose_validation_manifest"].endswith(
            "preflight/ntu_pose_validation_v1.json"
        )
        assert data["runtime_cache_mode"] == "required"
        assert data["runtime_cache_manifest"].endswith("runtime_cache_v1/manifest.json")


def test_ntu_training_uses_the_five_runtime_optimizations():
    config = load_experiment("confirm140_ntu_spikepose_frame")
    assert config["data"]["runtime_spatial_crops"] is True
    assert config["data"]["skeleton_cache_size"] >= 256
    assert config["training"]["target_heatmap_backend"] == "gpu"
    assert config["training"]["single_pass_validation"] is True
    assert config["training"]["firing_rate_batches"] == 1


def test_mam_uses_the_shared_tube_cache_and_single_pass_validation():
    config = load_experiment("confirm140_ntu_mamv2")
    assert config["data"]["runtime_cache_mode"] == "required"
    assert config["data"]["runtime_spatial_crops"] is True
    assert config["data"]["runtime_cache_root"].endswith("runtime_cache_v1")
    assert config["training"]["single_pass_validation"] is True


def test_paper_pilot_is_twenty_epochs():
    configs = plan_study("icassp2027_pilot20")
    assert len(configs) == 16
    assert all(config["run_type"] == "pilot" for config in configs)
    assert all(config["training"]["epochs"] == 20 for config in configs)
    assert all(
        config["training"].get("milestone_epochs") == [10, 20]
        for config in configs if config["training"]["epochs"] > 0
    )


def test_formal_matrix_respects_compute_cap():
    configs = plan_study("icassp2027_confirm140")
    assert max(config["training"]["epochs"] for config in configs) == 140


def test_native_ntu25_preview_is_retained():
    for config in plan_study("ntu25_preview8"):
        assert config["model"]["num_joints"] == 25
        assert config["data"]["joint_mapping"] == "ntu25_identity_v1"
        assert config["training"]["epochs"] == 8


def test_ntu_contiguous_clip_layout_resolves_setup_and_clip(tmp_path):
    dataset = NTUFrameDataset.__new__(NTUFrameDataset)
    dataset.extracted_frames_dir = tmp_path
    dataset.frame_layout = "setup_contiguous_clips"
    dataset.frame_clip_subdir = "contiguous_2x16"
    sample_id = "S013C001P001R001A001"
    frame_dir = (
        tmp_path / "S013" / "contiguous_2x16" / sample_id / "clip_01"
    )
    frame_dir.mkdir(parents=True)
    expected = frame_dir / "frame_000123.jpg"
    expected.touch()

    assert dataset._get_sample_frame_dir(sample_id) == frame_dir.parent
    assert dataset._get_frame_path(sample_id, 123) == expected
    second = frame_dir.parent / "clip_02" / "frame_000123.jpg"
    second.parent.mkdir()
    second.touch()
    assert dataset._get_frame_path(sample_id, 123, "clip_02") == second


def test_temporal_index_never_crosses_fixed_clip_boundary(tmp_path):
    dataset = NTUFrameDataset.__new__(NTUFrameDataset)
    sample_id = "S001C001P001R001A001"
    dataset.samples = [{
        "sample_id": sample_id, "rgb_frames": 100, "skeleton_frames": 100,
    }]
    dataset.skipped_samples = []
    dataset.extracted_frames_dir = tmp_path
    dataset.frame_layout = "setup_contiguous_clips"
    dataset.frame_clip_subdir = "contiguous_2x16"
    dataset.frame_stride = 1
    dataset.temporal_steps = 16
    dataset.temporal_frame_gap = 1
    dataset.minimum_temporal_history = 0
    dataset.exclude_overlapping_clips = True
    root = tmp_path / "S001" / "contiguous_2x16" / sample_id
    for clip, start in (("clip_01", 10), ("clip_02", 26)):
        directory = root / clip
        directory.mkdir(parents=True)
        for frame in range(start, start + 16):
            (directory / f"frame_{frame:06d}.jpg").touch()
    assert dataset._build_frame_index() == [
        (0, 25, "clip_01", 15), (0, 41, "clip_02", 15),
    ]


def test_clip_audit_requires_two_disjoint_consecutive_16_frame_clips(tmp_path):
    sequence = tmp_path / "S001" / "contiguous_2x16" / "S001C001P001R001A001"
    for clip, start in (("clip_01", 10), ("clip_02", 40)):
        directory = sequence / clip
        directory.mkdir(parents=True)
        for frame in range(start, start + 16):
            (directory / f"frame_{frame:06d}.jpg").touch()
    report = _clip_layout(tmp_path, "contiguous_2x16", 2, 16)
    assert report == {
        "sequences": 1,
        "audited_sequences": 1,
        "excluded_sequences_not_audited": 0,
        "valid_sequences": 1,
        "wrong_clip_count": 0,
        "wrong_clip_length": 0,
        "nonconsecutive_clips": 0,
        "overlapping_clips": 0,
        "overlapping_sample_ids": [],
    }
