from __future__ import annotations

import numpy as np

from spikepose_thesis.data.ntu.core.joint_mapping import joint_names_for_count


def sequence_groups(values: dict[str, np.ndarray]) -> list[np.ndarray]:
    explicit_identity = all(
        key in values for key in ("video_id", "clip_id", "frame_position_in_clip")
    )
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, (sample, person) in enumerate(zip(
        values["sample_id"], values["person_id"],
    )):
        key = (
            (
                str(values["video_id"][index]), str(person),
                str(values["clip_id"][index]),
            )
            if explicit_identity else (str(sample), str(person))
        )
        groups.setdefault(key, []).append(index)
    order = (
        values["frame_position_in_clip"]
        if explicit_identity else values["frame_index"]
    )
    return [
        np.asarray(sorted(indices, key=lambda item: int(order[item])))
        for indices in groups.values()
    ]


def contiguous_sequence_groups(values: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Split sequence-person rows at every non-consecutive frame boundary."""
    result = []
    for indices in sequence_groups(values):
        order = (
            values["frame_position_in_clip"][indices]
            if "frame_position_in_clip" in values
            else values["frame_index"][indices]
        ).astype(np.int64)
        result.extend(
            item for item in np.split(indices, np.flatnonzero(np.diff(order) != 1) + 1)
            if len(item)
        )
    return result


def _validated_temporal_groups(
    values: dict[str, np.ndarray], expected_length: int,
) -> tuple[list[np.ndarray], dict[str, int]]:
    required = {
        "video_id", "person_id", "clip_id", "frame_index",
        "frame_position_in_clip",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(
            "Strict temporal evaluation requires explicit clip identity; "
            f"missing {missing}"
        )
    valid = []
    drop_reasons: dict[str, int] = {}

    def drop(reason: str) -> None:
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    for indices in sequence_groups(values):
        positions = values["frame_position_in_clip"][indices].astype(np.int64)
        frames = values["frame_index"][indices].astype(np.int64)
        if len(indices) != expected_length:
            drop("unexpected_length")
        elif not np.array_equal(positions, np.arange(expected_length)):
            drop("invalid_frame_positions")
        elif len(np.unique(frames)) != expected_length:
            drop("duplicate_frame_index")
        elif not np.all(np.diff(frames) > 0):
            drop("non_increasing_frame_index")
        else:
            valid.append(indices)
    return valid, drop_reasons


def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
    valid = mask & np.isfinite(value)
    return float(value[valid].mean()) if valid.any() else float("nan")


def _nanmean(values) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _magnitude_ratio(
    prediction: np.ndarray, target: np.ndarray, mask: np.ndarray,
) -> float:
    epsilon = 1e-8
    prediction_sum = float(prediction[mask].sum())
    target_sum = float(target[mask].sum())
    if target_sum <= epsilon:
        return 1.0 if prediction_sum <= epsilon else float("inf")
    return prediction_sum / target_sum


def temporal_summary(
    values: dict[str, np.ndarray], *, strict: bool = False,
    expected_length: int | None = None,
) -> dict[str, object]:
    if strict:
        if expected_length is None or expected_length < 4:
            raise ValueError("Strict temporal evaluation requires expected_length >= 4")
        groups, drop_reasons = _validated_temporal_groups(values, expected_length)
    else:
        groups = contiguous_sequence_groups(values)
        drop_reasons = {}
    rows = []
    per_joint_rows: list[dict[str, np.ndarray]] = []
    peak_lags = []
    valid_velocity_pairs = 0
    valid_acceleration_triplets = 0
    velocity_intervals = 0
    acceleration_intervals = 0
    for indices in groups:
        if len(indices) < 4:
            drop_reasons["too_short_for_temporal_metrics"] = (
                drop_reasons.get("too_short_for_temporal_metrics", 0) + 1
            )
            continue
        pred = values["prediction"][indices]
        target = values["target"][indices]
        visible = values["visibility"][indices] > 0
        scale = values["scale"][indices]
        pred_v, target_v = np.diff(pred, axis=0), np.diff(target, axis=0)
        pred_a, target_a = np.diff(pred, n=2, axis=0), np.diff(target, n=2, axis=0)
        pred_j, target_j = np.diff(pred, n=3, axis=0), np.diff(target, n=3, axis=0)
        vel_mask = visible[1:] & visible[:-1]
        acc_mask = visible[2:] & visible[1:-1] & visible[:-2]
        jerk_mask = visible[3:] & visible[2:-1] & visible[1:-2] & visible[:-3]
        velocity_intervals += len(indices) - 1
        acceleration_intervals += len(indices) - 2
        valid_velocity_pairs += int(vel_mask.sum())
        valid_acceleration_triplets += int(acc_mask.sum())
        vel_error = np.linalg.norm(pred_v - target_v, axis=-1)
        acc_error = np.linalg.norm(pred_a - target_a, axis=-1)
        jerk_error = np.linalg.norm(pred_j - target_j, axis=-1)
        pred_vel_mag = np.linalg.norm(pred_v, axis=-1)
        target_vel_mag = np.linalg.norm(target_v, axis=-1)
        pred_acc_mag = np.linalg.norm(pred_a, axis=-1)
        target_acc_mag = np.linalg.norm(target_a, axis=-1)
        epsilon = 1e-8
        valid_motion = target_vel_mag[vel_mask]
        motion_threshold = (
            float(np.percentile(valid_motion, 75))
            if len(valid_motion) else float("inf")
        )
        high_motion = vel_mask & (target_vel_mag >= motion_threshold)
        spatial_error = np.linalg.norm(pred[1:] - target[1:], axis=-1)
        high_motion_pck = high_motion & (spatial_error <= 0.5 * scale[1:, None])
        joint_velocity = np.asarray([
            _masked_mean(vel_error[:, joint], vel_mask[:, joint])
            for joint in range(vel_error.shape[1])
        ])
        joint_acceleration = np.asarray([
            _masked_mean(acc_error[:, joint], acc_mask[:, joint])
            for joint in range(acc_error.shape[1])
        ])
        joint_jerk = np.asarray([
            _masked_mean(jerk_error[:, joint], jerk_mask[:, joint])
            for joint in range(jerk_error.shape[1])
        ])
        per_joint_rows.append({
            "velocity_error": joint_velocity,
            "acceleration_error": joint_acceleration,
            "jerk_error": joint_jerk,
        })
        for joint in range(pred_vel_mag.shape[1]):
            valid_velocity = vel_mask[:, joint]
            if valid_velocity.any():
                positions = np.flatnonzero(valid_velocity)
                pred_peak = positions[np.argmax(pred_vel_mag[positions, joint])]
                target_peak = positions[np.argmax(target_vel_mag[positions, joint])]
                peak_lags.append(float(pred_peak - target_peak))
        rows.append({
            "velocity_error": _masked_mean(vel_error, vel_mask),
            "acceleration_error": _masked_mean(acc_error, acc_mask),
            "acceleration_error_hb": _masked_mean(
                acc_error / scale[2:, None], acc_mask,
            ),
            "relative_acceleration_error": float(
                100.0 * acc_error[acc_mask].sum()
                / (target_acc_mag[acc_mask].sum() + epsilon)
            ),
            "acceleration_magnitude_ratio": _magnitude_ratio(
                pred_acc_mag, target_acc_mag, acc_mask,
            ),
            "velocity_magnitude_ratio": _magnitude_ratio(
                pred_vel_mag, target_vel_mag, vel_mask,
            ),
            "jerk_error": _masked_mean(jerk_error, jerk_mask),
            "high_motion_pck_0_5": float(
                high_motion_pck[high_motion].mean()
            ) if high_motion.any() else float("nan"),
            "high_motion_acceleration_error": _masked_mean(
                acc_error, acc_mask & high_motion[1:],
            ),
        })
    if not rows:
        if strict:
            raise ValueError(
                "Strict temporal evaluation produced zero valid sequences; "
                f"drop_reasons={drop_reasons}"
            )
        return {}
    summary: dict[str, object] = {
        key: _nanmean([row[key] for row in rows])
        for key in rows[0]
    }
    summary["peak_lag_frames"] = (
        float(np.mean(peak_lags)) if peak_lags else float("nan")
    )
    summary["peak_lag_absolute_frames"] = (
        float(np.mean(np.abs(peak_lags))) if peak_lags else float("nan")
    )
    summary["contiguous_segments"] = len(rows)
    summary["num_valid_sequences"] = len(rows)
    summary["num_dropped_sequences"] = int(sum(drop_reasons.values()))
    summary["drop_reasons"] = drop_reasons
    summary["num_velocity_intervals"] = velocity_intervals
    summary["num_acceleration_intervals"] = acceleration_intervals
    summary["num_valid_velocity_pairs"] = valid_velocity_pairs
    summary["num_valid_acceleration_triplets"] = valid_acceleration_triplets
    joint_names = joint_names_for_count(per_joint_rows[0]["velocity_error"].shape[0])
    summary["per_joint"] = {
        name: {
            metric: _nanmean([row[metric][joint] for row in per_joint_rows])
            for metric in per_joint_rows[0]
        }
        for joint, name in enumerate(joint_names)
    }
    summary.update({
        "vel_e": summary["velocity_error"],
        "acc_e": summary["acceleration_error"],
        "racc_e": summary["relative_acceleration_error"],
        "vmr": summary["velocity_magnitude_ratio"],
        "amr": summary["acceleration_magnitude_ratio"],
        "jerk_e": summary["jerk_error"],
        "hm_pck_hb_05": summary["high_motion_pck_0_5"],
        "hm_acc_e": summary["high_motion_acceleration_error"],
        "lag_frames": summary["peak_lag_frames"],
    })
    return summary
