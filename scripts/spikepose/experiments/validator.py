from __future__ import annotations

from ..models.registry import BLOCKS, HEADS, NECKS, NEURONS


def validate_config(config: dict) -> None:
    if not config.get("id") or not config.get("name"):
        raise ValueError("Every experiment requires canonical id and name")
    if config["experiment"] != config["id"]:
        raise ValueError("Experiment filename and canonical id must match")
    scope = config.get("dataset_scope")
    dataset = config.get("dataset")
    if scope is not None and dataset != scope:
        raise ValueError(
            f"Experiment {config['id']} is scoped to {scope}, not {dataset}"
        )
    model = config["model"]
    if int(model.get("num_steps", 0)) < 1:
        raise ValueError("model.num_steps must be at least one")
    temporal = model.get("temporal", {})
    strategy = temporal.get("input_strategy", "repeat")
    aggregation = temporal.get("aggregation", "mean")
    if strategy not in ("repeat", "previous_translate_current", "frames"):
        raise ValueError(
            "model.temporal.input_strategy must be repeat, "
            "previous_translate_current, or frames"
        )
    history_mode = temporal.get("history_mode", "real")
    if history_mode not in {
        "real", "repeat_current", "reverse", "batch_shuffle_history", "history_only",
    }:
        raise ValueError("Unsupported model.temporal.history_mode")
    state_mode = temporal.get("state_mode", "continuous")
    if state_mode not in {"continuous", "reset"}:
        raise ValueError("model.temporal.state_mode must be continuous or reset")
    snn_steps_per_frame = int(temporal.get("snn_steps_per_frame", 1))
    decouple_video_time = (
        bool(temporal.get("decouple_video_time", False)) or snn_steps_per_frame > 1
    )
    if snn_steps_per_frame < 1:
        raise ValueError("model.temporal.snn_steps_per_frame must be at least one")
    if decouple_video_time and strategy != "frames":
        raise ValueError("decoupled frame/SNN time requires input_strategy=frames")
    if decouple_video_time and state_mode != "reset":
        raise ValueError("decoupled frame/SNN time requires state_mode=reset")
    stage_steps = tuple(int(item) for item in temporal.get("stage_steps", ()))
    transform_kind = temporal.get("transform_kind", "none")
    output_steps = temporal.get("output_steps")
    if stage_steps:
        if strategy not in {"frames", "repeat"}:
            raise ValueError(
                "stage_steps requires frames or repeated-current input_strategy"
            )
        if len(stage_steps) != len(model["backbone"]["channels"]):
            raise ValueError("stage_steps must contain one value per backbone stage")
        if stage_steps[0] != int(model["num_steps"]):
            raise ValueError("stage_steps must start at model.num_steps")
        if any(left < right for left, right in zip(stage_steps, stage_steps[1:])):
            raise ValueError("stage_steps must be non-increasing")
        if max(stage_steps) > 4 or min(stage_steps) < 1:
            raise ValueError("cross-frame stage_steps must remain between 1 and 4")
        if transform_kind not in {"learned", "truncate"}:
            raise ValueError("stage_steps requires learned or truncate transform_kind")
        resolved_output = int(output_steps or stage_steps[-1])
        if not 1 <= resolved_output <= stage_steps[-1]:
            raise ValueError("output_steps must lie between 1 and the final stage steps")
    elif transform_kind != "none" or output_steps is not None:
        raise ValueError("transform_kind/output_steps require stage_steps")
    if state_mode == "reset" and stage_steps:
        raise ValueError("state_mode=reset cannot use stage_steps")
    early_stages = tuple(int(item) for item in temporal.get(
        "early_classifier_stages", (),
    ))
    early_weights = tuple(float(item) for item in temporal.get(
        "early_classifier_weights", (),
    ))
    available_stages = set(range(1, len(model["backbone"]["channels"]) + 1))
    if len(early_stages) != len(early_weights):
        raise ValueError("early classifier stages and weights must have equal length")
    if len(early_stages) != len(set(early_stages)):
        raise ValueError("early classifier stages must be unique")
    if not set(early_stages).issubset(available_stages):
        raise ValueError("early classifier contains an unknown backbone stage")
    if any(weight <= 0.0 for weight in early_weights) or sum(early_weights) >= 1.0:
        raise ValueError("early classifier weights must be positive and sum below one")
    refinement = {
        "residual_refinement", "feature_control_refinement",
        "heatmap_feedback", "heatmap_feedback_no_residual",
        "skeleton_heatmap_feedback",
    }
    joint_temporal = {
        "joint_temporal", "joint_temporal_spatial",
        "joint_confidence_temporal", "joint_motion_temporal",
        "joint_aligned_temporal", "joint_aligned_motion_temporal",
        "joint_aligned_confidence_temporal",
        "joint_residual_correction", "joint_difference_temporal",
        "joint_decoupled_space_time",
        "motion_trend_fixed", "motion_trend_gated",
    }
    if aggregation not in {"mean", "last", "heatmap_mean", "learned_heatmap",
                           *refinement, *joint_temporal}:
        raise ValueError(
            "Unsupported model.temporal.aggregation"
        )
    if (aggregation in {"heatmap_mean", "learned_heatmap", *refinement} and
            model["head"]["kind"] != "ann_heatmap"):
        raise ValueError(f"{aggregation} requires the ann_heatmap head")
    if aggregation == "learned_heatmap" and int(model["num_steps"]) > 2:
        raise ValueError("learned_heatmap supports at most two time steps")
    if aggregation in joint_temporal and model["head"]["kind"] != "linear_heatmap":
        raise ValueError(f"{aggregation} requires the linear_heatmap head")
    if aggregation in joint_temporal and int(model["num_steps"]) < 2:
        raise ValueError("joint-wise temporal aggregation requires at least two time steps")
    auxiliary_weight = float(temporal.get("intermediate_loss_weight", 0.0))
    if not 0.0 <= auxiliary_weight <= 1.0:
        raise ValueError("intermediate_loss_weight must lie between zero and one")
    if auxiliary_weight and aggregation not in {
        *refinement, "joint_aligned_temporal",
        "motion_trend_fixed", "motion_trend_gated",
    }:
        raise ValueError(
            "intermediate supervision requires refinement or aligned temporal aggregation"
        )
    if aggregation in {"motion_trend_fixed", "motion_trend_gated"}:
        if int(model["num_steps"]) != 4 or not decouple_video_time:
            raise ValueError(
                "motion trend aggregation requires four decoupled video frames"
            )
    kinematic_weight = float(temporal.get("kinematic_loss_weight", 0.0))
    if not 0.0 <= kinematic_weight <= 1.0:
        raise ValueError("kinematic_loss_weight must lie between zero and one")
    if kinematic_weight and aggregation != "joint_aligned_temporal":
        raise ValueError("kinematic supervision requires joint_aligned_temporal")
    if kinematic_weight and int(model["num_steps"]) < 3:
        raise ValueError("kinematic supervision requires at least three frames")
    if aggregation in refinement and int(model["num_steps"]) < 2:
        raise ValueError("temporal refinement requires at least two time steps")
    if aggregation in {
        "skeleton_heatmap_feedback", "joint_temporal_spatial",
        "joint_decoupled_space_time",
    }:
        edges = temporal.get("feedback_edges", [])
        joints = int(model["num_joints"])
        if not edges or any(len(edge) != 2 or min(edge) < 0 or max(edge) >= joints
                            for edge in edges):
            raise ValueError("skeleton feedback requires valid feedback_edges")
    radius = int(temporal.get("translation_pixels", 0))
    if radius < 0:
        raise ValueError("model.temporal.translation_pixels must be non-negative")
    if strategy == "previous_translate_current":
        if int(model["num_steps"]) != 2:
            raise ValueError("previous_translate_current requires model.num_steps=2")
        if radius < 1:
            raise ValueError("previous_translate_current requires translation_pixels >= 1")
    fixed = temporal.get("eval_translation")
    if fixed is not None and (len(fixed) != 2 or any(abs(int(v)) > radius for v in fixed)):
        raise ValueError("eval_translation must contain two offsets within translation_pixels")
    backbone = model["backbone"]
    if not (len(backbone["channels"]) == len(backbone["depths"]) ==
            len(backbone["blocks"])):
        raise ValueError("Backbone channels, depths and blocks must match")
    if any(item not in BLOCKS for item in backbone["blocks"]):
        raise ValueError(f"Supported blocks: {BLOCKS}")
    if model["neuron"]["kind"] not in NEURONS:
        raise ValueError(f"Supported neurons: {NEURONS}")
    if model["neck"]["kind"] not in NECKS:
        raise ValueError(f"Supported necks: {NECKS}")
    head = model["head"]
    if head["kind"] not in HEADS:
        raise ValueError(f"Supported heads: {HEADS}")
    if head["kind"] == "coordinate_classification":
        if int(head.get("split_ratio", 0)) < 1:
            raise ValueError("coordinate classification split_ratio must be at least one")
        if float(head.get("label_sigma", 0.0)) <= 0:
            raise ValueError("coordinate classification label_sigma must be positive")
    if head["kind"] == "coordinate_regression":
        if head.get("regression_variant") not in {"gap", "spatial", "integral"}:
            raise ValueError(
                "coordinate regression variant must be gap, spatial, or integral"
            )
    if model["neck"].get("interpolation", "nearest") not in ("nearest", "bilinear"):
        raise ValueError("neck.interpolation must be nearest or bilinear")
    if model["head"].get("upsample_factor", 2) not in (1, 2):
        raise ValueError("head.upsample_factor must be 1 or 2")
    if model["head"].get("output_init", "small_normal") not in (
            "kaiming", "small_normal"):
        raise ValueError("head.output_init must be kaiming or small_normal")
    stages = set(range(1, len(backbone["channels"]) + 1))
    if not set(backbone["output_stages"]).issubset(stages):
        raise ValueError("output_stages contains an unknown stage")
    if not set(backbone.get("attention_stages", ())).issubset(stages):
        raise ValueError("attention_stages contains an unknown stage")
    mask = model["neck"].get("stage_mask")
    if model["neck"]["kind"] == "masked_add":
        outputs = backbone["output_stages"]
        if mask is None or len(mask) != len(outputs):
            raise ValueError("masked_add requires one stage_mask value per output stage")
        if any(value not in (0, 1) for value in mask) or not any(mask):
            raise ValueError("stage_mask must be binary and enable at least one stage")
