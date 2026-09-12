from __future__ import annotations

from pathlib import Path
import secrets


# Experiment/config IDs remain unchanged.  Only their on-disk path components
# are canonicalized so existing YAML files and checkpoints stay compatible.
_MPII_CATEGORIES = {
    "main_route": "architecture_search",
    "backbone_fpn": "backbone_comparison",
    "energy_repair": "energy_measurement",
    "temporal": "temporal_ablation",
    "training_profile": "training_profile_ablation",
    "training_extension": "training_schedule",
    "all": "summaries",
    "heatmap_decoding": "heatmap_decoding_ablation",
    "stage_fusion": "stage_fusion",
}

_MPII_BATCHES = {
    ("architecture_search", "mpii_main_route_20ep"): "preliminary_v1_ep020",
    ("backbone_comparison", "mpii_bf_20ep"): "v1_ep020",
    ("energy_measurement", "measurement_v2"): "v2",
    ("pose_ablation", "mpii_pose_20ep"): "v1_ep020",
    ("temporal_ablation", "mpii_temporal_20ep"): "v1_ep020",
    ("training_profile_ablation", "profile_ablation_v1_ep020"): "profile_a_to_b",
    ("training_schedule", "constant_1e6"): "constant_lr_1e-6_ep210",
    ("training_schedule", "cosine_restart_1e5"): "cosine_restart_lr_1e-5_ep210",
    ("training_schedule", "scratch_cosine_210"): "cosine_scratch_ep210",
    ("heatmap_decoding_ablation", "heatmap_v1"): "v1",
    ("heatmap_decoding_ablation", "heatmap_v1_smoke"): "v1_smoke",
}

_MPII_RUNS = {
    ("stage_fusion", "profile_b_seed42", "s0"):
        ("stage_fusion", "profile_b", "s0_stage4_only"),
    ("stage_fusion", "profile_b_seed42", "s1"):
        ("stage_fusion", "profile_b", "s1_stages34"),
    ("stage_fusion", "profile_b_seed42", "s2"):
        ("stage_fusion", "profile_b", "s2_stages234"),
    ("stage_fusion", "profile_b_seed42", "s3"):
        ("stage_fusion", "profile_b", "s3_stages1234"),
    ("stage_fusion", "profile_b_seed42", "s4"):
        ("stage_fusion", "profile_b", "s4_without_stage2"),
    ("stage_fusion", "profile_b_seed42", "s5"):
        ("stage_fusion", "profile_b", "s5_without_stage3"),
    ("stage_fusion", "profile_b_seed42", "s6"):
        ("stage_fusion", "profile_b", "s6_without_stage4"),
    ("pose_ablation", "mpii_pose_20ep", "baseline"):
        ("architecture_evolution", "profile_b", "b0_original_baseline"),
    ("main_route", "evolution_profile_b", "e0"):
        ("architecture_evolution", "profile_b", "e0_unified_baseline"),
    ("main_route", "evolution_profile_b", "e1"):
        ("architecture_evolution", "profile_b", "e1_single_scale_head"),
    ("main_route", "evolution_profile_b", "e2"):
        ("architecture_evolution", "profile_b", "e2_stage123_fusion"),
    ("main_route", "mpii_main_route_20ep", "four_stage"):
        ("architecture_evolution", "profile_b", "e3_four_stage_fusion"),
    ("main_route", "mpii_main_route_20ep", "mem_ann"):
        ("architecture_evolution", "profile_b", "e4_membrane_residual"),
    ("main_route", "mpii_main_route_20ep", "mem_spikefpn_annhead"):
        ("architecture_evolution", "profile_b", "e5_spike_fpn_ann_head"),
    ("backbone_fpn", "mpii_bf_20ep", "mem_fpn"):
        ("architecture_evolution", "profile_b", "e6_spike_fpn_linear_head"),
    ("main_route", "mpii_main_route_20ep", "resformer_s3"):
        ("architecture_evolution", "profile_b", "e7_resformer_stage3"),
    ("main_route", "mpii_main_route_20ep", "resformer_s4"):
        ("architecture_evolution", "profile_b", "e8_resformer_stage4"),
    ("backbone_fpn", "mpii_bf_20ep", "resformer_fpn"):
        ("architecture_evolution", "profile_b", "e9_resformer_stages34"),
    # The complete main-route launcher passes one shared category/batch to all
    # eleven configs. Accept that form as an alias for the historical locations.
    ("main_route", "mpii_main_route_20ep", "baseline"):
        ("architecture_evolution", "profile_b", "b0_original_baseline"),
    ("main_route", "mpii_main_route_20ep", "e0"):
        ("architecture_evolution", "profile_b", "e0_unified_baseline"),
    ("main_route", "mpii_main_route_20ep", "e1"):
        ("architecture_evolution", "profile_b", "e1_single_scale_head"),
    ("main_route", "mpii_main_route_20ep", "e2"):
        ("architecture_evolution", "profile_b", "e2_stage123_fusion"),
    ("main_route", "mpii_main_route_20ep", "mem_fpn"):
        ("architecture_evolution", "profile_b", "e6_spike_fpn_linear_head"),
    ("main_route", "mpii_main_route_20ep", "resformer_fpn"):
        ("architecture_evolution", "profile_b", "e9_resformer_stages34"),
}

_MPII_EXPERIMENTS = {
    ("architecture_search", "preliminary_v1_ep020", "head1x"): "single_head",
    ("backbone_comparison", "v1_ep020", "multiconv_fpn"): "multiscale_conv_fpn",
    ("backbone_comparison", "v1_ep020", "sew_fpn"): "sew_resnet18_fpn",
    ("energy_measurement", "v2", "mem_spikefpn_annhead"): "membrane_spike_fpn_ann_head",
    ("energy_measurement", "v2", "multiconv_fpn"): "multiscale_conv_fpn",
    ("pose_ablation", "v1_ep020", "add"): "fusion_add",
    ("pose_ablation", "v1_ep020", "lif"): "activation_lif",
    ("pose_ablation", "v1_ep020", "relu"): "activation_relu",
    ("pose_ablation", "v1_ep020", "stage1"): "stage_1",
    ("pose_ablation", "v1_ep020", "stage12"): "stages_1_2",
    ("pose_ablation", "v1_ep020", "stage123"): "stages_1_2_3",
    ("training_profile_ablation", "profile_a_to_b", "p0"): "p0_profile_a_baseline",
    ("training_profile_ablation", "profile_a_to_b", "p1"): "p1_small_normal_init",
    ("training_profile_ablation", "profile_a_to_b", "p2"): "p2_warmup_cosine_schedule",
    ("training_profile_ablation", "profile_a_to_b", "p3"): "p3_gradient_clipping",
    ("training_profile_ablation", "profile_a_to_b", "p4"): "p4_scale_augmentation",
    ("training_profile_ablation", "profile_a_to_b", "p5"): "p5_profile_b_full",
}

def canonical_output_parts(
    dataset: str, category: str, batch_name: str, experiment: str,
) -> tuple[str, str, str]:
    """Return canonical path slugs without changing experiment/config IDs."""
    safe_experiment = experiment.replace("/", "_")
    if dataset != "mpii":
        return category, batch_name, safe_experiment
    run_route = _MPII_RUNS.get((category, batch_name, safe_experiment))
    if run_route is not None:
        return run_route
    if category == "main_route" and batch_name == "evolution_flip_comparison":
        return "architecture_evolution", "comparisons/flip_test", safe_experiment
    canonical_category = _MPII_CATEGORIES.get(category, category)
    canonical_batch = _MPII_BATCHES.get(
        (canonical_category, batch_name), batch_name,
    )
    canonical_experiment = _MPII_EXPERIMENTS.get(
        (canonical_category, canonical_batch, safe_experiment),
        safe_experiment,
    )
    return canonical_category, canonical_batch, canonical_experiment


def output_batch_dir(
    root: Path, dataset: str, category: str, batch_name: str,
) -> Path:
    if dataset == "mpii" and category == "main_route" and batch_name in {
        "mpii_main_route_20ep", "evolution_profile_b",
    }:
        return root / dataset / "architecture_evolution" / "profile_b"
    if (dataset == "mpii" and category == "stage_fusion" and
            batch_name == "profile_b_seed42"):
        return root / dataset / "stage_fusion" / "profile_b"
    canonical_category, canonical_batch, _ = canonical_output_parts(
        dataset, category, batch_name, "_batch_placeholder_",
    )
    return root / dataset / canonical_category / canonical_batch


def output_run_dir(
    root: Path, dataset: str, category: str, batch_name: str,
    experiment: str, seed: int,
) -> Path:
    canonical_category, canonical_batch, canonical_experiment = (
        canonical_output_parts(dataset, category, batch_name, experiment)
    )
    return (
        root / dataset / canonical_category / canonical_batch /
        canonical_experiment / f"seed_{seed}"
    )


def validate_seed_targets(
    root: Path, dataset: str, category: str, batch_name: str,
    experiments: list[str] | tuple[str, ...], seeds: list[int],
    resume: bool,
) -> list[Path]:
    """Reject duplicate seeds, accidental overwrites, and invalid resumes."""
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {seeds}")
    targets = [
        output_run_dir(
            root, dataset, category, batch_name, experiment, seed,
        )
        for experiment in experiments
        for seed in seeds
    ]
    if resume:
        missing = [
            path / "checkpoints" / "last.pt" for path in targets
            if not (path / "checkpoints" / "last.pt").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Cannot resume; missing checkpoint(s): "
                + ", ".join(str(path) for path in missing)
            )
    else:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing seed run(s): "
                + ", ".join(str(path) for path in existing)
            )
    return targets


def existing_seed_values(root: Path) -> set[int]:
    """Collect recorded integer seed directory names below an output root."""
    values = set()
    if not root.exists():
        return values
    for path in root.rglob("seed_*"):
        if not path.is_dir():
            continue
        suffix = path.name[len("seed_"):]
        if suffix.isdigit():
            values.add(int(suffix))
    return values


def generate_random_seeds(root: Path, count: int) -> list[int]:
    """Generate unique positive 31-bit seeds not recorded under root."""
    if count < 1:
        raise ValueError("Random seed count must be at least 1")
    maximum = 2**31 - 1
    used = existing_seed_values(root)
    if count > maximum - len(used):
        raise ValueError("Not enough unused 31-bit seed values remain")
    generated = []
    selected = set()
    while len(generated) < count:
        seed = secrets.randbelow(maximum) + 1
        if seed in used or seed in selected:
            continue
        selected.add(seed)
        generated.append(seed)
    return generated
