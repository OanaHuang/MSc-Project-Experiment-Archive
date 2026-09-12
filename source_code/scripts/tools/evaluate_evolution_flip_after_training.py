from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.spikepose.artifacts import output_batch_dir, output_run_dir


OUTPUT_ROOT = PROJECT_ROOT / "Outputs_New"
COMPARISON_ROOT = output_batch_dir(
    OUTPUT_ROOT, "mpii", "main_route", "evolution_flip_comparison",
)

RUNS = {
    "B0": output_run_dir(OUTPUT_ROOT, "mpii", "pose_ablation", "mpii_pose_20ep", "baseline", 42),
    "E0": output_run_dir(OUTPUT_ROOT, "mpii", "main_route", "evolution_profile_b", "e0", 42),
    "E1": output_run_dir(OUTPUT_ROOT, "mpii", "main_route", "evolution_profile_b", "e1", 42),
    "E2": output_run_dir(OUTPUT_ROOT, "mpii", "main_route", "evolution_profile_b", "e2", 42),
    "E3": output_run_dir(OUTPUT_ROOT, "mpii", "main_route", "mpii_main_route_20ep", "four_stage", 42),
    "E4": output_run_dir(OUTPUT_ROOT, "mpii", "main_route", "mpii_main_route_20ep", "mem_ann", 42),
    "E5": output_run_dir(OUTPUT_ROOT, "mpii", "main_route", "mpii_main_route_20ep", "mem_spikefpn_annhead", 42),
    "E6": output_run_dir(OUTPUT_ROOT, "mpii", "backbone_fpn", "mpii_bf_20ep", "mem_fpn", 42),
    "E7": output_run_dir(OUTPUT_ROOT, "mpii", "main_route", "mpii_main_route_20ep", "resformer_s3", 42),
    "E8": output_run_dir(OUTPUT_ROOT, "mpii", "main_route", "mpii_main_route_20ep", "resformer_s4", 42),
    "E9": output_run_dir(OUTPUT_ROOT, "mpii", "backbone_fpn", "mpii_bf_20ep", "resformer_fpn", 42),
}

WAIT_FOR_COMPLETION = ("E0", "E1", "E2")
POLL_SECONDS = 60


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def profile_b_training_complete() -> bool:
    for experiment in WAIT_FOR_COMPLETION:
        status_path = RUNS[experiment] / "status.json"
        if not status_path.is_file():
            return False
        status = read_json(status_path)
        if status.get("status") != "completed":
            return False
        if not (RUNS[experiment] / "checkpoints" / "best.pt").is_file():
            return False
    return True


def wait_for_profile_b() -> None:
    while not profile_b_training_complete():
        time.sleep(POLL_SECONDS)


def validate_inputs() -> None:
    problems = []
    for experiment, run_dir in RUNS.items():
        for relative in (
            Path("config/resolved.yaml"),
            Path("checkpoints/best.pt"),
            Path("metrics/dual_pckh_summary.json"),
        ):
            if not (run_dir / relative).is_file():
                problems.append(f"{experiment}: missing {relative}")
    if problems:
        raise RuntimeError("Evolution evaluation inputs are incomplete: " + "; ".join(problems))


def flip_result_is_complete(run_dir: Path) -> bool:
    summary_path = run_dir / "metrics_flip_test" / "dual_pckh_summary.json"
    if not summary_path.is_file():
        return False
    summary = read_json(summary_path)
    return bool(summary.get("flip_test")) and int(summary.get("samples", 0)) == 2925


def evaluate_flip(experiment: str, run_dir: Path) -> None:
    if flip_result_is_complete(run_dir):
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env.get("FLIP_PHYSICAL_GPU", "1")
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "MPII" / "evaluate.py"),
        "--run", str(run_dir),
        "--device", "cuda:0",
        "--flip-test",
        "--output-dir", str(run_dir / "metrics_flip_test"),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
    if not flip_result_is_complete(run_dir):
        raise RuntimeError(f"{experiment}: Flip Test output failed validation")


def grouped_columns(prefix: str, summary: dict) -> dict:
    grouped = summary.get("official_grouped_pckh", {})
    return {
        f"{prefix}_{joint}_pckh": grouped.get(joint)
        for joint in ("head", "shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    }


def build_comparison() -> list[dict]:
    rows = []
    for experiment, run_dir in RUNS.items():
        no_flip = read_json(run_dir / "metrics" / "dual_pckh_summary.json")
        flip = read_json(run_dir / "metrics_flip_test" / "dual_pckh_summary.json")
        no_flip_pckh = float(no_flip["official_pckh"])
        flip_pckh = float(flip["official_pckh"])
        row = {
            "experiment": experiment,
            "model_name": no_flip.get("model_name", no_flip.get("model")),
            "samples": int(no_flip["samples"]),
            "no_flip_official_pckh": no_flip_pckh,
            "flip_official_pckh": flip_pckh,
            "flip_gain_pp": 100.0 * (flip_pckh - no_flip_pckh),
            "no_flip_pckhn": float(no_flip["custom_pckh"]),
            "flip_pckhn": float(flip["custom_pckh"]),
            "pckhn_gain_pp": 100.0 * (float(flip["custom_pckh"]) - float(no_flip["custom_pckh"])),
            "no_flip_metrics_path": str(run_dir / "metrics"),
            "flip_metrics_path": str(run_dir / "metrics_flip_test"),
        }
        row.update(grouped_columns("no_flip", no_flip))
        row.update(grouped_columns("flip", flip))
        rows.append(row)
    return rows


def save_comparison(rows: list[dict]) -> None:
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = COMPARISON_ROOT / "evolution_flip_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (COMPARISON_ROOT / "evolution_flip_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8",
    )
    (COMPARISON_ROOT / "status.json").write_text(
        json.dumps({"status": "completed", "experiments": list(RUNS)}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    (COMPARISON_ROOT / "status.json").write_text(
        json.dumps({"status": "waiting_for_profile_b", "experiments": list(RUNS)}, indent=2),
        encoding="utf-8",
    )
    wait_for_profile_b()
    validate_inputs()
    (COMPARISON_ROOT / "status.json").write_text(
        json.dumps({"status": "evaluating_flip", "experiments": list(RUNS)}, indent=2),
        encoding="utf-8",
    )
    for experiment, run_dir in RUNS.items():
        evaluate_flip(experiment, run_dir)
    save_comparison(build_comparison())


if __name__ == "__main__":
    main()
