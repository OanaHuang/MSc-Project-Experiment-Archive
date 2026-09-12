from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any

import yaml


CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
NTU_SETUP_PATTERN = re.compile(r"^S(0[0-1][0-9]|0[2][0])$")

# Runtime setup overrides are allowed to expand a setup-scoped Pilot (for
# example the S010 MAM study) to the canonical full split.  These fields
# describe the split and its integrity gates, so retaining their scoped values
# would silently keep training on the smaller metadata even when
# ``--ntu-setups full`` is requested.
NTU_FULL_DATA_FIELDS = (
    "train_metadata",
    "validation_metadata",
    "test_metadata",
    "exclusion_metadata",
    "exclude_overlapping_clips",
    "pose_validation_mode",
    "pose_validation_manifest",
    "skeleton_cache_size",
    "validation_fraction",
    "validation_strategy",
    "validation_performers",
    "exclusion_policy",
    "expected_raw_sequences",
    "expected_quality_exclusions",
    "expected_usable_sequences",
)


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {path}")
    return value


def deep_merge(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _named(kind: str, name: str) -> dict:
    path = CONFIG_ROOT / kind / f"{name}.yaml"
    value = read_yaml(path)
    parent = value.pop("extends", None)
    return deep_merge(_named(kind, parent), value) if parent else value


def _temporal_model_overrides(spec: dict) -> dict:
    video_frames = int(spec["video_frames"])
    updates_per_frame = int(spec["snn_updates_per_frame"])
    input_mode = str(spec["input_mode"])
    if video_frames < 1 or updates_per_frame < 1:
        raise ValueError("video_frames and snn_updates_per_frame must be positive")
    if input_mode not in {"repeat_current", "real_history", "stream", "scheduled"}:
        raise ValueError(f"Unsupported two-clock input mode: {input_mode}")
    if input_mode == "repeat_current" and video_frames != 1:
        raise ValueError("repeat_current requires video_frames=1")
    frames_input = input_mode in {"real_history", "stream", "scheduled"}
    schedule: tuple[int, ...] = ()
    if input_mode == "scheduled":
        raw_schedule = spec.get("update_schedule")
        if not isinstance(raw_schedule, (list, tuple)) or not raw_schedule:
            raise ValueError("scheduled input requires a non-empty update_schedule")
        schedule = tuple(int(index) for index in raw_schedule)
        expected = tuple(
            frame
            for frame in range(video_frames)
            for _ in range(updates_per_frame)
        )
        if schedule != expected:
            raise ValueError(
                "scheduled updates must be causal, ordered and balanced; "
                f"expected {list(expected)}, got {list(schedule)}"
            )
        num_steps = len(schedule)
    else:
        num_steps = video_frames if frames_input else updates_per_frame
    result = {
        "num_steps": num_steps,
        "temporal": {
            "video_frames": video_frames,
            "input_strategy": (
                "scheduled_frames" if input_mode == "scheduled"
                else "frames" if frames_input else "repeat"
            ),
            "aggregation": str(spec["readout"]),
            "history_mode": "real",
            "state_mode": str(spec.get("state_policy", "reset")),
            "snn_steps_per_frame": updates_per_frame,
            "decouple_video_time": bool(
                frames_input and input_mode != "scheduled" and updates_per_frame > 1
            ),
            "update_schedule": schedule,
        },
    }
    if frames_input:
        result["neck"] = {"kind": "spike_fpn_temporal"}
    return result


def _apply_profile(config: dict, profile_name: str | None) -> dict:
    if profile_name is None:
        return config
    profile = _named("profiles", profile_name)
    if profile.get("id") != profile_name:
        raise ValueError(f"Profile id must match filename: {profile_name}")
    result = deepcopy(config)
    result["profile"] = profile_name
    result["run_type"] = str(profile["run_type"])
    prefix = str(profile.get("paper_id_prefix", ""))
    if prefix and not str(result["paper_id"]).startswith(prefix):
        result["paper_id"] = prefix + str(result["paper_id"])

    training = result["training"]
    epochs = int(training.get("epochs", 0))
    profile_training = profile["training"]
    target_epochs = int(profile_training["epochs"])
    if epochs > 0:
        training["epochs"] = target_epochs
        phases = []
        for phase in training.get("phases", []):
            start = int(phase["start_epoch"])
            if start > target_epochs:
                continue
            phases.append({
                **phase,
                "end_epoch": min(int(phase["end_epoch"]), target_epochs),
            })
        if "phases" in training:
            training["phases"] = phases
        for key, value in profile_training.items():
            if key != "epochs":
                training[key] = deepcopy(value)

    source = result.get("initialization", {}).get("source")
    source_overrides = profile.get("source_overrides", {})
    if source in source_overrides:
        result["initialization"]["source"] = source_overrides[source]

    if result.get("action") == "refinement":
        refinement = profile["refinement_protocol"]
        result["refinement_protocol"] = deep_merge(
            result.get("refinement_protocol", {}), refinement,
        )
    if result.get("priority") == "conditional":
        result["priority"] = "pilot"
    return result


def load_experiment(name: str, profile: str | None = None) -> dict:
    matches = list((CONFIG_ROOT / "experiments").glob(f"**/{name}.yaml"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one experiment named {name!r}; found {len(matches)}"
        )
    experiment = read_yaml(matches[0])
    if experiment.get("id") != name:
        raise ValueError(f"Experiment id must match filename: {matches[0]}")
    model_name = experiment.pop("model_config")
    protocol_name = experiment.pop("protocol")
    resolved = deep_merge(_named("protocols", protocol_name), _named("models", model_name))
    temporal = experiment.pop("temporal", None)
    resolved = deep_merge(resolved, experiment)
    if temporal:
        resolved["temporal"] = temporal
        resolved["model"] = deep_merge(
            resolved["model"], _temporal_model_overrides(temporal),
        )
    resolved["model_config"] = model_name
    resolved["protocol"] = protocol_name
    resolved.setdefault("run_type", "formal")
    resolved = _apply_profile(resolved, profile)
    validate_experiment(resolved)
    return resolved


def parse_ntu_setups(value: str) -> tuple[str, ...] | None:
    """Parse `full`, a range such as S010-S015, or a comma-separated list."""
    text = str(value).strip().upper()
    if text == "FULL":
        return None
    setups: list[str] = []
    for token in (item.strip() for item in text.split(",")):
        if not token:
            raise ValueError("NTU setup selection contains an empty item")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(NTU_SETUP_PATTERN.fullmatch(x) for x in parts):
                raise ValueError(f"Invalid NTU setup range: {token}")
            start, end = (int(x[1:]) for x in parts)
            if start > end:
                raise ValueError(f"NTU setup range is reversed: {token}")
            setups.extend(f"S{index:03d}" for index in range(start, end + 1))
        elif NTU_SETUP_PATTERN.fullmatch(token):
            setups.append(token)
        else:
            raise ValueError(f"Invalid NTU setup: {token}")
    result = tuple(dict.fromkeys(setups))
    if not result:
        raise ValueError("At least one NTU setup is required")
    return result


def apply_pilot_runtime_overrides(
    config: dict, *, epochs: int | None = None, ntu_setups: str | None = None,
) -> dict:
    """Return an isolated Pilot config with safe runtime-only overrides."""
    if epochs is None and ntu_setups is None:
        return config
    if config.get("run_type") != "pilot":
        raise ValueError("Runtime epoch/setup overrides are restricted to Pilot runs")
    result = deepcopy(config)
    variant_parts: list[str] = []
    if epochs is not None:
        target = int(epochs)
        if target < 1:
            raise ValueError("--epochs must be positive")
        result["training"]["epochs"] = target
        milestones = [value for value in (10, 20, 60) if value <= target]
        if target not in milestones:
            milestones.append(target)
        result["training"]["milestone_epochs"] = sorted(set(milestones))
        phases = []
        for phase in result["training"].get("phases", []):
            if int(phase["start_epoch"]) <= target:
                phases.append({
                    **phase, "end_epoch": min(int(phase["end_epoch"]), target),
                })
        if "phases" in result["training"]:
            result["training"]["phases"] = phases
    is_ntu = str(result.get("dataset", "")).startswith("ntu")
    if ntu_setups is not None and not is_ntu:
        raise ValueError("--ntu-setups is only valid for NTU Pilot experiments")
    if is_ntu:
        canonical_protocol = (
            "ntu_cv80" if result["dataset"] == "ntu60_cv" else "ntu_cs80"
        )
        canonical_data = _named("protocols", canonical_protocol)["data"]
        for key in NTU_FULL_DATA_FIELDS:
            if key in canonical_data:
                result["data"][key] = deepcopy(canonical_data[key])
            else:
                result["data"].pop(key, None)
        setup_selection = ntu_setups if ntu_setups is not None else "full"
        setups = parse_ntu_setups(setup_selection)
        result["data"]["setup_filter"] = list(setups) if setups is not None else None
        setup_label = "full" if setups is None else "_".join(setups)
        variant_parts.append(f"setups_{setup_label}")
    elif epochs is not None:
        variant_parts.append("runtime_epochs")
    result["runtime_overrides"] = {
        "epochs": epochs,
        "ntu_setups": ntu_setups if ntu_setups is not None else ("full" if is_ntu else None),
    }
    result["run_variant"] = "__".join(variant_parts)
    validate_experiment(result)
    return result


def load_study(name: str) -> dict:
    value = _named("studies", name)
    if value.get("id") != name:
        raise ValueError(f"Study id must match filename: {name}")
    experiments = []
    for included in value.get("includes", []):
        experiments.extend(load_study(included)["experiments"])
    experiments.extend(value.get("experiments", []))
    value["experiments"] = list(dict.fromkeys(experiments))
    return value


def experiment_names() -> list[str]:
    return sorted(path.stem for path in (CONFIG_ROOT / "experiments").glob("**/*.yaml"))


def validate_experiment(config: dict) -> None:
    required = {"id", "paper_id", "stage", "dataset", "model", "training"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"{config.get('id', '<unknown>')} missing fields: {sorted(missing)}")
    if config["dataset"] not in {"mpii", "ntu60_cs", "ntu60_cv"}:
        raise ValueError(f"Unsupported dataset protocol: {config['dataset']}")
    seeds = config["training"].get("seeds", [])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError(f"Experiment {config['id']} requires unique training seeds")
    initialization = config.get("initialization", {"mode": "scratch"})
    initialization_mode = initialization.get("mode", "scratch")
    if initialization_mode not in {"scratch", "checkpoint", "official_checkpoint"}:
        raise ValueError(
            f"Experiment {config['id']} has unknown initialization mode "
            f"{initialization_mode!r}"
        )
    if initialization_mode == "checkpoint" and not initialization.get("source"):
        raise ValueError(f"Experiment {config['id']} has no checkpoint source")
    if initialization_mode == "official_checkpoint" and not initialization.get("path"):
        raise ValueError(f"Experiment {config['id']} has no official checkpoint path")
    if (
        initialization.get("allow_incomplete_source", False)
        and config.get("run_type", "formal") != "pilot"
    ):
        raise ValueError(
            f"Experiment {config['id']} may only use an incomplete source in a Pilot run"
        )
