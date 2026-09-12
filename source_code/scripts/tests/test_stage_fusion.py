from __future__ import annotations

from pathlib import Path
import unittest

from scripts.spikepose.experiments import resolve_config, validate_config
from scripts.spikepose.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "scripts" / "experiments"
TASK_PATH = PROJECT_ROOT / "scripts" / "MPII" / "configs" / "task.yaml"
TRAINING_PATH = PROJECT_ROOT / "scripts" / "MPII" / "configs" / "training.yaml"


def resolve(experiment: str) -> dict:
    config = resolve_config(
        EXPERIMENT_ROOT, experiment, TASK_PATH, TRAINING_PATH,
        {"training": {"seed": 42, "epochs": 120}},
    )
    validate_config(config)
    return config


class StageFusionTests(unittest.TestCase):
    def test_masks_follow_the_declared_s_series(self) -> None:
        expected = {
            "s0": [0, 0, 0, 1],
            "s1": [0, 0, 1, 1],
            "s2": [0, 1, 1, 1],
            "s3": [1, 1, 1, 1],
            "s4": [1, 0, 1, 1],
            "s5": [1, 1, 0, 1],
            "s6": [1, 1, 1, 0],
        }
        for experiment, mask in expected.items():
            config = resolve(experiment)
            self.assertEqual(config["model"]["neck"]["stage_mask"], mask)
            self.assertEqual(config["model"]["backbone"]["output_stages"], [1, 2, 3, 4])

    def test_all_variants_have_identical_parameter_capacity(self) -> None:
        counts = []
        for experiment in ("s0", "s1", "s2", "s3", "s4", "s5", "s6"):
            model = build_model(resolve(experiment))
            counts.append(sum(parameter.numel() for parameter in model.parameters()))
            self.assertEqual(set(model.neck.projections), {"1", "2", "3", "4"})
        self.assertEqual(len(set(counts)), 1)


if __name__ == "__main__":
    unittest.main()
