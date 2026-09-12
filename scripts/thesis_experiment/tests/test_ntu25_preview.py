from pathlib import Path

import numpy as np
import torch

from spikepose_thesis.core.config import load_experiment
from spikepose_thesis.data.ntu.core.config import NTU_FLIP_PAIRS, NTU_JOINT_NAMES
from spikepose_thesis.data.ntu.core.joint_mapping import (
    NTU25_TO_MPII16,
    joint_layout,
)
from spikepose_thesis.evaluation.runner import summarize_predictions
from spikepose_thesis.models import build_model
from spikepose_thesis.training.checkpoint import load_mpii16_to_ntu25_model


def test_native_ntu25_layout_contract() -> None:
    layout = joint_layout("ntu25_identity_v1")
    assert layout["num_joints"] == 25
    assert layout["joint_names"] == NTU_JOINT_NAMES
    assert layout["flip_pairs"] == NTU_FLIP_PAIRS
    assert (layout["head_index"], layout["neck_index"]) == (3, 2)
    assert layout["map_to_mpii16"] is False


def test_preview_configs_use_native_outputs_and_staged_warm_start() -> None:
    for name, head in (
        ("pilot8_ntu25_framewise", "head"),
        ("pilot8_ntu25_hrnet_w32", "final_layer"),
    ):
        config = load_experiment(name)
        assert config["model"]["num_joints"] == 25
        assert config["data"]["joint_mapping"] == "ntu25_identity_v1"
        assert config["initialization"]["load"] == "mpii16_to_ntu25"
        assert config["training"]["epochs"] == 8
        assert config["training"]["phases"][0]["train_modules"] == [head]
        assert config["training"]["phases"][0]["end_epoch"] == 3
    assert load_experiment("pilot8_ntu25_hrnet_w32")["initialization"][
        "allow_incomplete_source"
    ] is True


def _assert_transplant(source_name: str, target_name: str, key: str,
                       checkpoint: Path) -> None:
    source = build_model(load_experiment(source_name))
    with torch.no_grad():
        value = source.state_dict()[key]
        for joint in range(16):
            value[joint].fill_(float(joint + 1))
    torch.save({"model_state_dict": source.state_dict()}, checkpoint)
    target = build_model(load_experiment(target_name))
    original = target.state_dict()[key].clone()
    _, report = load_mpii16_to_ntu25_model(checkpoint, target, "cpu")
    result = target.state_dict()[key]
    for source_joint, target_joint in enumerate(NTU25_TO_MPII16.tolist()):
        assert torch.equal(
            result[target_joint],
            torch.full_like(result[target_joint], float(source_joint + 1)),
        )
    fresh = sorted(set(range(25)) - set(NTU25_TO_MPII16.tolist()))
    assert torch.equal(result[fresh], original[fresh])
    assert key in report["remapped_output_keys"]
    assert report["fresh_target_joint_indices"] == fresh


def test_framewise_and_hrnet_output_heads_are_transplanted(tmp_path: Path) -> None:
    _assert_transplant(
        "mamv2_fullcs_p00_source", "pilot8_ntu25_framewise",
        "head.output.weight", tmp_path / "framewise.pt",
    )
    _assert_transplant(
        "pilot20_ntu_hrnet_w32", "pilot8_ntu25_hrnet_w32",
        "final_layer.weight", tmp_path / "hrnet.pt",
    )


def test_prediction_summary_labels_all_25_native_joints() -> None:
    samples = 2
    values = {
        "prediction": np.zeros((samples, 25, 2), dtype=np.float32),
        "target": np.zeros((samples, 25, 2), dtype=np.float32),
        "visibility": np.ones((samples, 25), dtype=np.float32),
        "scale": np.ones(samples, dtype=np.float32),
        "sample_id": np.asarray(["a", "b"]),
        "frame_index": np.asarray([0, 0]),
        "person_id": np.asarray(["primary", "primary"]),
    }
    summary = summarize_predictions(values)
    assert tuple(summary["per_joint"]) == NTU_JOINT_NAMES
