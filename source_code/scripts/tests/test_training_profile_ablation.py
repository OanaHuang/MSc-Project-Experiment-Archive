from __future__ import annotations

from pathlib import Path
import unittest

from scripts.spikepose.experiments import load_ablation, resolve_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "scripts" / "experiments"
ABLATION_ROOT = PROJECT_ROOT / "scripts" / "ablations"
TASK_PATH = PROJECT_ROOT / "scripts" / "MPII" / "configs" / "task.yaml"
TRAINING_PATH = PROJECT_ROOT / "scripts" / "MPII" / "configs" / "training.yaml"


def resolve(experiment: str) -> dict:
    return resolve_config(
        EXPERIMENT_ROOT, experiment, TASK_PATH, TRAINING_PATH,
    )


class TrainingProfileAblationTests(unittest.TestCase):
    def test_group_contains_the_ordered_p_series(self) -> None:
        group = load_ablation(ABLATION_ROOT, "training_profile")
        self.assertEqual(
            group["experiments"], ["p0", "p1", "p2", "p3", "p4", "p5"],
        )

    def test_all_variants_keep_the_b0_topology(self) -> None:
        reference = resolve("p0")["model"]
        for experiment in ("p2", "p3", "p4", "p5"):
            candidate = resolve(experiment)["model"]
            candidate["head"]["output_init"] = reference["head"]["output_init"]
            self.assertEqual(candidate, reference)

    def test_p1_changes_only_final_output_initialization(self) -> None:
        p0 = resolve("p0")["model"]
        p1 = resolve("p1")["model"]
        self.assertEqual(p0["head"]["output_init"], "kaiming")
        self.assertEqual(p1["head"]["output_init"], "small_normal")
        p1["head"]["output_init"] = p0["head"]["output_init"]
        self.assertEqual(p1, p0)

    def test_training_interventions_are_added_in_order(self) -> None:
        p1 = resolve("p1")["training"]
        p2 = resolve("p2")["training"]
        p3 = resolve("p3")["training"]
        p4 = resolve("p4")["training"]
        p5 = resolve("p5")["training"]

        self.assertEqual(p1["scheduler"]["kind"], "plateau")
        self.assertEqual(p2["scheduler"]["kind"], "warmup_cosine")
        self.assertEqual(p2["learning_rate"], 0.0003)
        self.assertIsNone(p2["gradient_clip"])
        self.assertEqual(p2["augmentation"]["rotation_degrees"], 0.0)
        self.assertIsNone(p2["augmentation"]["scale_range"])

        self.assertEqual(p3["gradient_clip"], 5.0)
        self.assertIsNone(p3["augmentation"]["scale_range"])

        self.assertEqual(p4["augmentation"]["scale_range"], [0.75, 1.25])
        self.assertEqual(p4["augmentation"]["rotation_degrees"], 0.0)

        self.assertEqual(p5["augmentation"]["scale_range"], [0.75, 1.25])
        self.assertEqual(p5["augmentation"]["rotation_degrees"], 30.0)


if __name__ == "__main__":
    unittest.main()
