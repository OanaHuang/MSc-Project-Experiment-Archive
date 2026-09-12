from pathlib import Path

import numpy as np
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.experiments import plan_study
from spikepose_thesis.models import build_model
from spikepose_thesis.models.baselines import load_official_checkpoint
from spikepose_thesis.refinement.filters import ema_shared
from spikepose_thesis.refinement.runner import _source_prediction_path
from spikepose_thesis.training.checkpoint import load_spatial_model
from spikepose_thesis.training.runner import _configure_trainable
from tools.run_icassp2027_queue import _parse_gpu_occupancy


PAPER_SEEDS = [42, 365198782, 2071461405]


def test_queue_maps_compute_processes_to_physical_gpu_indices() -> None:
    inventory = "0, GPU-zero\n1, GPU-one\n2, GPU-two\n3, GPU-three\n"
    compute = "GPU-zero, 123\nGPU-one, 456\n"
    discovered, busy = _parse_gpu_occupancy(inventory, compute)
    assert discovered == {0, 1, 2, 3}
    assert busy == {0, 1}


def test_final_pilot_and_confirm_rosters_are_isolated() -> None:
    pilot = plan_study("icassp2027_pilot20")
    confirm = plan_study("icassp2027_confirm140")
    assert len(pilot) == 16
    assert len(confirm) == 16
    assert all(item["run_type"] == "pilot" for item in pilot)
    assert all(item["training"]["epochs"] == 20 for item in pilot)
    assert all(item["run_type"] == "formal" for item in confirm)
    assert all(item["training"]["epochs"] == 140 for item in confirm)
    assert all(
        not str(item.get("initialization", {}).get("source", "")).startswith(
            "pilot20_"
        )
        for item in confirm
    )


def test_final_seed_policy_matches_the_paper_plan() -> None:
    confirm = {item["id"]: item for item in plan_study("icassp2027_confirm140")}
    for name in (
        "confirm140_m_ann_s3", "confirm140_m_s3_u1",
        "confirm140_m_s3_u2", "confirm140_ntu_spikepose_frame",
        "confirm140_ntu_mamv2", "confirm140_ntu_simplebaseline_r50",
        "confirm140_ntu_hrnet_w32",
    ):
        assert confirm[name]["training"]["seeds"] == PAPER_SEEDS
    for name in (
        "confirm140_ntu_mamv2_noalign",
        "confirm140_ntu_mamv2_fixedleak",
        "confirm140_ntu_mamv2_noresidual",
        "confirm140_factor_v1u1", "confirm140_factor_v1u2",
        "confirm140_factor_v1u4", "confirm140_factor_v2u2",
        "confirm140_factor_v2u4", "confirm140_factor_v4u4",
    ):
        assert confirm[name]["training"]["seeds"] == [42]


def test_no_residual_fusion_ablation_disables_only_the_output_skip() -> None:
    for name, profile in (
        ("pilot20_mamv2_noresidual", "pilot20"),
        ("confirm140_ntu_mamv2_noresidual", None),
    ):
        temporal = load_experiment(name, profile=profile)["model"]["temporal"]
        assert temporal["memory_use_current_direct_path"] is False
        assert temporal["memory_use_residual_offset"] is True
        assert temporal["memory_use_dynamic_gate"] is True


def test_table5_component_ablation_flags_are_isolated() -> None:
    expected = {
        "pilot20_mamv2_nomotiontoken": (False, False, True),
        "pilot20_mamv2_nooffsethead": (True, False, True),
        "pilot20_mamv2_nomemoryupdate": (True, True, False),
    }
    for name, flags in expected.items():
        temporal = load_experiment(name, profile="pilot20")["model"]["temporal"]
        assert (
            temporal["memory_use_motion_token"],
            temporal["memory_use_residual_offset"],
            temporal["memory_use_memory_update"],
        ) == flags
        assert temporal["memory_use_current_direct_path"] is True
        assert temporal["memory_use_dynamic_gate"] is True

    study = plan_study("mam_components_pilot20")
    assert len(study) == 7
    assert {item["id"] for item in study}.issuperset(expected)


def test_final_mam_phase_selectors_resolve() -> None:
    for name in (
        "confirm140_ntu_mamv2", "confirm140_ntu_mamv2_noalign",
        "confirm140_ntu_mamv2_fixedleak",
        "confirm140_ntu_mamv2_noresidual",
        "pilot20_mamv2_nomotiontoken",
        "pilot20_mamv2_nooffsethead",
        "pilot20_mamv2_nomemoryupdate",
    ):
        profile = "pilot20" if name.startswith("pilot20_") else None
        config = load_experiment(name, profile=profile)
        model = build_model(config)
        for phase in config["training"]["phases"]:
            optimizer = _configure_trainable(model, config["training"], phase)
            assert optimizer.param_groups


def test_controls_reuse_the_frame_checkpoint_without_temporal_keys(
    tmp_path: Path,
) -> None:
    source = build_model(load_experiment("confirm140_ntu_spikepose_frame"))
    checkpoint = tmp_path / "source.pt"
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    for name in (
        "confirm140_control_v1u1", "confirm140_control_v1u2",
        "confirm140_control_v2u2",
    ):
        target = build_model(load_experiment(name))
        _, report = load_spatial_model(checkpoint, target, "cpu")
        assert report["fresh_target_temporal_keys"] == 0
        assert report["loaded_spatial_keys"] > 0


def test_official_baselines_are_direct_ntu_finetunes() -> None:
    expected = {
        "confirm140_ntu_simplebaseline_r50": (
            "pose_resnet",
            "Datasets/MPII/official_simplebaseline/pose_resnet_50_256x256.pth.tar",
        ),
        "confirm140_ntu_hrnet_w32": (
            "hrnet",
            "Datasets/MPII/official_simplebaseline/pose_hrnet_w32_256x256.pth",
        ),
    }
    for name, (family, path) in expected.items():
        config = load_experiment(name)
        assert config["model"]["family"] == family
        assert config["initialization"] == {
            "mode": "official_checkpoint",
            "path": path,
            "format": "official_hrnet",
        }


def test_official_checkpoint_loader_is_strict(tmp_path: Path) -> None:
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    checkpoint = tmp_path / "official.pth"
    torch.save(source.state_dict(), checkpoint)
    report = load_official_checkpoint(checkpoint, target, "cpu")
    assert report["loaded_tensors"] == len(target.state_dict())
    assert report["tensor_sha256"] == report["source_tensor_sha256"]
    assert report["aligned_tensor_sha256"] == report["source_tensor_sha256"]
    for source_value, target_value in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(source_value, target_value)


def test_eval_studies_register_only_the_expected_trained_refiner() -> None:
    pilot = {item["id"]: item for item in plan_study("icassp2027_pilot_eval")}
    confirm = plan_study("icassp2027_confirm_eval")
    assert len(pilot) == 7
    assert len(confirm) == 6
    assert pilot["pilot20_refine_jointwise"]["training"]["epochs"] == 20
    assert pilot["pilot20_refine_jointwise"]["training"]["seeds"] == [42]
    assert all(
        config["training"]["epochs"] == 0
        for name, config in pilot.items() if name != "pilot20_refine_jointwise"
    )
    assert all(config["training"]["epochs"] == 0 for config in confirm)


def test_jointwise_refiner_uses_the_full_per_frame_source_archive(tmp_path) -> None:
    config = load_experiment("pilot20_refine_jointwise", profile="pilot20")
    assert config["initialization"]["source"] == "mamv2_fullcs_p00_source"
    assert config["refinement"] == {
        "method": "jtr_jointwise", "causal": True,
        "look_ahead": 0, "joints": 16,
    }
    path = _source_prediction_path(tmp_path, "train", "jtr_jointwise")
    assert path.name == "predictions_per_frame.npz"


def test_shared_ema_is_causal() -> None:
    value = np.asarray([0.0, 2.0, 4.0], dtype=np.float32).reshape(3, 1, 1)
    filtered = ema_shared(value, 0.5)
    np.testing.assert_allclose(filtered[:, 0, 0], [0.0, 1.0, 2.5])
