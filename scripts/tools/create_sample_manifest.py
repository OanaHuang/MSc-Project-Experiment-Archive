from __future__ import annotations

from pathlib import Path

from scripts.MPII.datasets import MPIIPoseDataset
from scripts.NTU_RGBD.datasets import build_dataset
from scripts.spikepose.experiments import experiment_roots, resolve_config
from scripts.spikepose.visualization import create_manifest


def create_batch_manifest(project_root: Path, dataset: str, output_path: Path,
                          selection_seed: int) -> Path:
    dataset_dir = "MPII" if dataset == "mpii" else "NTU_RGBD"
    config = resolve_config(
        experiment_roots(project_root, dataset), "baseline",
        project_root / "scripts" / dataset_dir / "configs" / "task.yaml",
        project_root / "scripts" / dataset_dir / "configs" / "training.yaml",
    )
    if dataset == "mpii":
        data = config["data"]
        split = MPIIPoseDataset(
            project_root / data["visualization_metadata"],
            project_root / data["images_dir"], data["image_size"],
            data["heatmap_size"], data["sigma"], data["crop_expansion"], False,
        )
    else:
        split = build_dataset(project_root, config, "validation")
    visual = config["visualization"]
    create_manifest(split, output_path, selection_seed,
                    visual["first_count"], visual["random_count"])
    return output_path
