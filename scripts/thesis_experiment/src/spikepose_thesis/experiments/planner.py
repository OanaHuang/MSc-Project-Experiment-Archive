from __future__ import annotations

import ast
from pathlib import Path

import torch

from spikepose_thesis.core.config import (
    CONFIG_ROOT, experiment_names, load_experiment, load_study,
)
from spikepose_thesis.models import build_model


ICASSP2027_PILOT_TRAIN_IDS = {
    "pilot20_m_ann_s3", "pilot20_m_s3_u1", "pilot20_m_s3_u2",
    "pilot20_factor_v1u1", "pilot20_factor_v1u2",
    "pilot20_factor_v1u4", "pilot20_factor_v2u2",
    "pilot20_factor_v2u4", "pilot20_factor_v4u4",
    "mamv2_fullcs_p00_source", "mamv2_fullcs20",
    "pilot20_ntu_simplebaseline_r50", "pilot20_ntu_hrnet_w32",
    "mamv2_p06_noalign", "mamv2_p07_fixed_decay",
    "pilot20_mamv2_noresidual",
}
ICASSP2027_CONFIRM_TRAIN_IDS = {
    "confirm140_m_ann_s3", "confirm140_m_s3_u1", "confirm140_m_s3_u2",
    "confirm140_factor_v1u1", "confirm140_factor_v1u2",
    "confirm140_factor_v1u4", "confirm140_factor_v2u2",
    "confirm140_factor_v2u4", "confirm140_factor_v4u4",
    "confirm140_ntu_spikepose_frame", "confirm140_ntu_mamv2",
    "confirm140_ntu_simplebaseline_r50", "confirm140_ntu_hrnet_w32",
    "confirm140_ntu_mamv2_noalign", "confirm140_ntu_mamv2_fixedleak",
    "confirm140_ntu_mamv2_noresidual",
}
ICASSP2027_PILOT_EVAL_IDS = {
    "pilot20_control_v1u1", "pilot20_control_v1u2",
    "pilot20_control_v2u2", "pilot20_eval_ema_shared",
    "pilot20_refine_jointwise", "pilot20_eval_sg_causal",
    "pilot20_eval_one_euro",
}
ICASSP2027_CONFIRM_EVAL_IDS = {
    "confirm140_control_v1u1", "confirm140_control_v1u2",
    "confirm140_control_v2u2", "confirm140_eval_ema_shared",
    "confirm140_eval_sg_causal", "confirm140_eval_one_euro",
}
NTU25_PREVIEW_IDS = {
    "pilot8_ntu25_framewise", "pilot8_ntu25_hrnet_w32",
}
PAPER_REQUIRED_IDS = (
    ICASSP2027_PILOT_TRAIN_IDS
    | ICASSP2027_CONFIRM_TRAIN_IDS
    | ICASSP2027_PILOT_EVAL_IDS
    | ICASSP2027_CONFIRM_EVAL_IDS
    | NTU25_PREVIEW_IDS
)


def _ordered(names: list[str], profile: str | None = None) -> list[str]:
    remaining = list(dict.fromkeys(names))
    selected: list[str] = []
    while remaining:
        progressed = False
        for name in list(remaining):
            source = load_experiment(
                name, profile=profile,
            ).get("initialization", {}).get("source")
            if not source or source not in remaining:
                selected.append(name)
                remaining.remove(name)
                progressed = True
        if not progressed:
            raise ValueError(f"Experiment dependency cycle: {remaining}")
    return selected


def plan_study(name: str) -> list[dict]:
    study = load_study(name)
    profile = study.get("profile")
    return [
        load_experiment(item, profile=profile)
        for item in _ordered(study["experiments"], profile)
    ]


def _forbidden_imports(package_root: Path) -> list[str]:
    failures = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("Scripts", "scripts")):
                        failures.append(f"{path}:{node.lineno}: {alias.name}")
            if module and module.startswith(("Scripts", "scripts")):
                failures.append(f"{path}:{node.lineno}: {module}")
    return failures


def validate_repository(forward: bool = False) -> dict:
    names = experiment_names()
    missing = PAPER_REQUIRED_IDS - set(names)
    if missing:
        raise ValueError(f"Missing paper experiments: {sorted(missing)}")
    configs = {name: load_experiment(name) for name in names}
    for name, config in configs.items():
        source = config.get("initialization", {}).get("source")
        if source and source not in configs:
            raise ValueError(f"{name} references unknown source {source}")
        if source and config.get("initialization", {}).get("match_seed"):
            missing_seeds = set(config["training"]["seeds"]) - set(
                configs[source]["training"]["seeds"]
            )
            if missing_seeds:
                raise ValueError(f"{name} lacks same-seed source runs: {missing_seeds}")
    over_budget = {
        name: int(config["training"]["epochs"])
        for name, config in configs.items()
        if config.get("run_type", "formal") == "formal"
        and int(config["training"]["epochs"]) > 140
    }
    if over_budget:
        raise ValueError(f"Formal experiments exceed the 140-epoch cap: {over_budget}")
    paper_pilot = plan_study("icassp2027_pilot20")
    if {config["id"] for config in paper_pilot} != ICASSP2027_PILOT_TRAIN_IDS:
        raise ValueError("icassp2027_pilot20 does not match the frozen paper roster")
    if any(config.get("run_type") != "pilot" for config in paper_pilot):
        raise ValueError("Every final-paper Pilot20 run must stay in the Pilot root")
    if any(int(config["training"]["epochs"]) != 20 for config in paper_pilot):
        raise ValueError("Every final-paper Pilot training run must use 20 epochs")

    paper_confirm = plan_study("icassp2027_confirm140")
    if {config["id"] for config in paper_confirm} != ICASSP2027_CONFIRM_TRAIN_IDS:
        raise ValueError("icassp2027_confirm140 does not match the frozen paper roster")
    if any(config.get("run_type") != "formal" for config in paper_confirm):
        raise ValueError("Every Confirm140 run must stay in the formal output root")
    if any(int(config["training"]["epochs"]) != 140 for config in paper_confirm):
        raise ValueError("Every Confirm140 training run must use 140 epochs")
    for config in paper_confirm:
        source = config.get("initialization", {}).get("source")
        if source and configs[source].get("run_type") != "formal":
            raise ValueError(
                f"Confirm140 run {config['id']} points to Pilot source {source}"
            )

    for study_name, expected in (
        ("icassp2027_pilot_eval", ICASSP2027_PILOT_EVAL_IDS),
        ("icassp2027_confirm_eval", ICASSP2027_CONFIRM_EVAL_IDS),
    ):
        evaluation = plan_study(study_name)
        if {config["id"] for config in evaluation} != expected:
            raise ValueError(f"{study_name} does not match the frozen eval roster")
        trained_refiners = {
            config["id"]: int(config["training"]["epochs"])
            for config in evaluation
            if int(config["training"]["epochs"]) != 0
        }
        expected_trained = (
            {"pilot20_refine_jointwise": 20}
            if study_name == "icassp2027_pilot_eval" else {}
        )
        if trained_refiners != expected_trained:
            raise ValueError(
                f"{study_name} has unexpected trained refiners: {trained_refiners}"
            )

    ntu25_preview = plan_study("ntu25_preview8")
    if {config["id"] for config in ntu25_preview} != NTU25_PREVIEW_IDS:
        raise ValueError("ntu25_preview8 does not match the retained NTU25 roster")
    for config in ntu25_preview:
        if config.get("run_type") != "pilot":
            raise ValueError(f"NTU25 preview {config['id']} must remain a Pilot")
        if int(config["training"]["epochs"]) != 8:
            raise ValueError(f"NTU25 preview {config['id']} must remain 8 epochs")
        if int(config["model"]["num_joints"]) != 25:
            raise ValueError(f"NTU25 preview {config['id']} must predict 25 joints")
        if config["data"].get("joint_mapping") != "ntu25_identity_v1":
            raise ValueError(f"NTU25 preview {config['id']} must use native NTU25 joints")
    failures = _forbidden_imports(Path(__file__).resolve().parents[1])
    if failures:
        raise ValueError("Legacy imports found:\n" + "\n".join(failures))
    forward_models = 0
    if forward:
        seen = set()
        for config in configs.values():
            signature = repr(config["model"])
            if signature in seen:
                continue
            seen.add(signature)
            model = build_model(config).eval()
            temporal = config["model"].get("temporal", {})
            if temporal.get("input_strategy") in {"frames", "scheduled_frames"}:
                image = torch.zeros(
                    1, int(temporal.get("video_frames", config["model"]["num_steps"])),
                    3, 64, 64,
                )
            else:
                image = torch.zeros(1, 3, 64, 64)
            with torch.no_grad():
                prediction = model(image)
            expected_joints = int(config["model"].get("num_joints", 16))
            if tuple(prediction.shape[:2]) != (1, expected_joints):
                raise ValueError(f"Bad forward output for {config['id']}: {prediction.shape}")
            forward_models += 1
    return {
        "experiments": len(configs), "required_experiments": len(PAPER_REQUIRED_IDS),
        "forward_models": forward_models, "legacy_imports": 0,
        "joint_layout_target": "ntu18_pending_mapping",
        "retained_auxiliary_layout": "ntu25_identity_v1",
        "config_root": str(CONFIG_ROOT),
    }
