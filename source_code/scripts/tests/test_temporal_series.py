from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.spikepose.experiments import (
    ablation_roots, experiment_roots, load_ablation, resolve_config,
    validate_config,
)
from scripts.spikepose.models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOTS = experiment_roots(PROJECT_ROOT, "mpii")
ABLATION_ROOTS = ablation_roots(PROJECT_ROOT, "mpii")
TASK_PATH = PROJECT_ROOT / "scripts" / "MPII" / "configs" / "task.yaml"
TRAINING_PATH = PROJECT_ROOT / "scripts" / "MPII" / "configs" / "training.yaml"


def resolve(experiment: str) -> dict:
    config = resolve_config(
        EXPERIMENT_ROOTS, experiment, TASK_PATH, TRAINING_PATH,
        {"training": {"seed": 42, "epochs": 120}},
    )
    validate_config(config)
    return config


class TemporalSeriesTests(unittest.TestCase):
    def test_minimum_history_aligns_t_series_targets(self) -> None:
        from scripts.NTU_RGBD.datasets.ntu_frame_dataset import NTUFrameDataset

        with tempfile.TemporaryDirectory() as directory:
            sample_id = "S010C001P007R001A001"
            frame_dir = Path(directory) / sample_id
            frame_dir.mkdir()
            for index in range(10, 30):
                (frame_dir / f"frame_{index:06d}.jpg").touch()

            def targets(steps: int, gap: int) -> list[int]:
                dataset = object.__new__(NTUFrameDataset)
                dataset.samples = [{
                    "sample_id": sample_id,
                    "rgb_frames": 30,
                    "skeleton_frames": 30,
                }]
                dataset.extracted_frames_dir = Path(directory)
                dataset.frame_stride = 1
                dataset.temporal_steps = steps
                dataset.temporal_frame_gap = gap
                dataset.minimum_temporal_history = 3
                dataset.skipped_samples = []
                return [frame for _, frame in dataset._build_frame_index()]

            expected = list(range(13, 30))
            self.assertEqual(targets(1, 1), expected)
            self.assertEqual(targets(2, 1), expected)
            self.assertEqual(targets(3, 1), expected)
            self.assertEqual(targets(4, 1), expected)
            self.assertEqual(targets(2, 5), list(range(15, 30)))

    def test_only_current_frame_builds_supervision_targets(self) -> None:
        from scripts.NTU_RGBD.datasets.ntu_frame_dataset import NTUFrameDataset

        dataset = object.__new__(NTUFrameDataset)
        dataset.frame_index = [(0, 15)]
        dataset.temporal_steps = 4
        dataset.temporal_frame_gap = 5
        dataset.return_temporal_sequence = True
        dataset.transform = None
        calls = []

        def get_item(sample_index, frame, parameters, include_target=True):
            calls.append((sample_index, frame, include_target))
            item = {"image": torch.full((3, 2, 2), float(frame))}
            if include_target:
                item["heatmaps"] = torch.ones(1, 2, 2)
            return item

        dataset._get_frame_item = get_item
        output = dataset[0]

        self.assertEqual(calls, [
            (0, 0, False), (0, 5, False), (0, 10, False), (0, 15, True),
        ])
        self.assertEqual(tuple(output["image"].shape), (4, 3, 2, 2))
        self.assertIn("heatmaps", output)
        self.assertEqual(output["temporal_frame_indices"].tolist(), [0, 5, 10, 15])

    def test_group_contains_the_ordered_e0_temporal_series(self) -> None:
        group = load_ablation(ABLATION_ROOTS, "temporal")
        self.assertEqual(group["dataset_scope"], "mpii")
        self.assertEqual(group["reference"], "e0")
        self.assertEqual(
            group["experiments"],
            ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
             "t9", "t10", "t11", "t12", "t13", "t14", "t15", "t16",
             "t17", "t18", "t19"],
        )

    def test_all_variants_keep_e0_training_conditions(self) -> None:
        reference = resolve("e0")
        for experiment in ("t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8",
                           "t9", "t10", "t11", "t12", "t13", "t14", "t15",
                           "t16", "t17", "t18", "t19"):
            candidate = resolve(experiment)
            self.assertEqual(candidate["training"], reference["training"])
            self.assertEqual(candidate["data"], reference["data"])
            model = deepcopy(candidate["model"])
            baseline = deepcopy(reference["model"])
            model.pop("num_steps")
            model.pop("temporal")
            model.pop("neuron")
            baseline.pop("num_steps")
            baseline.pop("temporal")
            baseline.pop("neuron")
            self.assertEqual(model, baseline)

    def test_historical_e0_and_e6_resolve_without_architecture_drift(self) -> None:
        e0 = resolve("e0")["model"]
        self.assertEqual(e0["num_steps"], 2)
        self.assertEqual(e0["backbone"]["channels"], [64, 128, 256])
        self.assertEqual(e0["backbone"]["depths"], [1, 2, 2])
        self.assertEqual(e0["backbone"]["output_stages"], [2, 3])
        self.assertEqual(e0["neck"]["kind"], "concat")
        self.assertEqual(e0["head"]["kind"], "ann_heatmap")
        self.assertEqual(e0["head"]["output_init"], "small_normal")

        e6 = resolve("mem_fpn")["model"]
        self.assertEqual(e6["num_steps"], 2)
        self.assertEqual(e6["backbone"]["channels"], [64, 128, 256, 256])
        self.assertEqual(e6["backbone"]["depths"], [1, 2, 2, 2])
        self.assertEqual(e6["backbone"]["output_stages"], [1, 2, 3, 4])
        self.assertEqual(e6["neck"]["kind"], "spike_fpn")
        self.assertEqual(e6["head"]["kind"], "linear_heatmap")

    def test_declared_temporal_interventions(self) -> None:
        expected = {
            "t0": (2, "repeat", "mean", 0),
            "t1": (1, "repeat", "last", 0),
            "t2": (2, "repeat", "last", 0),
            "t3": (2, "previous_translate_current", "last", 2),
            "t4": (2, "previous_translate_current", "last", 4),
            "t5": (2, "previous_translate_current", "last", 8),
            "t6": (2, "previous_translate_current", "mean", 4),
            "t7": (2, "repeat", "heatmap_mean", 0),
            "t8": (2, "repeat", "learned_heatmap", 0),
            "t9": (2, "repeat", "mean", 0),
            "t10": (2, "repeat", "mean", 0),
            "t11": (2, "repeat", "mean", 0),
            "t12": (2, "repeat", "mean", 0),
            "t13": (2, "repeat", "residual_refinement", 0),
            "t14": (2, "repeat", "feature_control_refinement", 0),
            "t15": (2, "repeat", "heatmap_feedback", 0),
            "t16": (2, "repeat", "heatmap_feedback_no_residual", 0),
            "t17": (2, "repeat", "heatmap_feedback", 0),
            "t18": (3, "repeat", "heatmap_feedback", 0),
            "t19": (2, "repeat", "skeleton_heatmap_feedback", 0),
        }
        for experiment, values in expected.items():
            model = resolve(experiment)["model"]
            temporal = model["temporal"]
            self.assertEqual(
                (model["num_steps"], temporal["input_strategy"],
                 temporal["aggregation"], temporal["translation_pixels"]),
                values,
            )

    def test_translated_sequence_keeps_current_image_as_last_step(self) -> None:
        config = resolve("t4")
        config["model"]["backbone"] = {
            "channels": [8], "depths": [1], "blocks": ["conv"],
            "output_stages": [1], "attention_stages": [],
        }
        config["model"]["neck"]["out_channels"] = 8
        config["model"]["head"]["hidden_channels"] = 8
        model = build_model(config).eval()
        image = torch.randn(2, 3, 32, 32)
        sequence = model._make_sequence(image)
        self.assertEqual(tuple(sequence.shape), (2, 2, 3, 32, 32))
        self.assertTrue(torch.equal(sequence[-1], image))
        self.assertFalse(torch.equal(sequence[0], image))
        output = model(image)
        self.assertEqual(tuple(output.shape), (2, 16, 64, 64))

    def test_post_head_temporal_readouts_preserve_shape_and_capacity(self) -> None:
        reference_parameters = None
        for experiment in ("t0", "t7", "t8"):
            config = resolve(experiment)
            config["model"]["backbone"] = {
                "channels": [8], "depths": [1], "blocks": ["conv"],
                "output_stages": [1], "attention_stages": [],
            }
            config["model"]["neck"]["out_channels"] = 8
            config["model"]["head"]["hidden_channels"] = 8
            model = build_model(config).eval()
            output = model(torch.randn(2, 3, 32, 32))
            self.assertEqual(tuple(output.shape), (2, 16, 64, 64))
            parameters = sum(item.numel() for item in model.parameters())
            if experiment == "t0":
                reference_parameters = parameters
            elif experiment == "t7":
                self.assertEqual(parameters, reference_parameters)
            else:
                self.assertEqual(parameters, reference_parameters + 2)

    def test_learned_heatmap_starts_as_heatmap_mean(self) -> None:
        torch.manual_seed(7)
        common = dict(in_channels=4, hidden_channels=4, num_joints=3,
                      output_size=(8, 8), upsample_factor=1)
        from scripts.spikepose.models.heads import HeatmapHead
        mean_head = HeatmapHead(**common, aggregation="heatmap_mean").eval()
        learned_head = HeatmapHead(**common, aggregation="learned_heatmap").eval()
        learned_head.load_state_dict(mean_head.state_dict(), strict=False)
        value = torch.randn(2, 2, 4, 8, 8)
        self.assertTrue(torch.allclose(mean_head(value), learned_head(value), atol=1e-6))

    def test_snn_native_variants_change_only_neuron_dynamics(self) -> None:
        reference = resolve("t0")
        expected = {
            "t9": (True, False, False),
            "t10": (False, True, False),
            "t11": (False, False, True),
            "t12": (True, True, False),
        }
        for experiment, flags in expected.items():
            candidate = resolve(experiment)
            reference_model = deepcopy(reference["model"])
            candidate_model = deepcopy(candidate["model"])
            reference_neuron = reference_model.pop("neuron")
            candidate_neuron = candidate_model.pop("neuron")
            self.assertEqual(candidate_model, reference_model)
            self.assertEqual(
                (candidate_neuron.get("membrane_readout", False),
                 candidate_neuron.get("learnable_decay", False),
                 candidate_neuron.get("learnable_initial_membrane", False)),
                flags,
            )
            for key in ("kind", "decay", "threshold", "max_spikes"):
                self.assertEqual(candidate_neuron[key], reference_neuron[key])

    def test_snn_native_neuron_parameters_preserve_e0_initial_state(self) -> None:
        from scripts.spikepose.models.neurons import MultiStepILIF
        value = torch.tensor([[[[[0.4]]]], [[[[0.4]]]]])
        baseline = MultiStepILIF()
        membrane = MultiStepILIF(membrane_readout=True)
        decay = MultiStepILIF(learnable_decay=True)
        initial = MultiStepILIF(learnable_initial_membrane=True)
        self.assertTrue(torch.equal(baseline(value), decay(value)))
        self.assertTrue(torch.equal(baseline(value), initial(value)))
        self.assertAlmostEqual(float(decay.current_decay().detach()), 0.9, places=6)
        self.assertGreater(float(membrane(value)[0, 0, 0, 0, 0]), 0.0)
        self.assertAlmostEqual(
            float(membrane.membrane_readout_logit.sigmoid().detach()), 0.01,
            places=6,
        )

    def test_refinement_variants_return_final_and_per_step_heatmaps(self) -> None:
        for experiment in ("t13", "t14", "t15", "t16", "t17", "t18", "t19"):
            config = resolve(experiment)
            config["model"]["backbone"] = {
                "channels": [16], "depths": [1], "blocks": ["conv"],
                "output_stages": [1], "attention_stages": [],
            }
            config["model"]["neck"]["out_channels"] = 16
            config["model"]["head"]["hidden_channels"] = 8
            model = build_model(config).eval()
            image = torch.randn(2, 3, 32, 32)
            final, temporal = model(image, return_intermediates=True)
            steps = config["model"]["num_steps"]
            self.assertEqual(len(temporal), steps)
            self.assertEqual(tuple(final.shape), (2, 16, 64, 64))
            self.assertTrue(torch.equal(final, temporal[-1]))

    def test_refinement_controls_have_expected_capacity(self) -> None:
        counts = {}
        for experiment in ("t13", "t14", "t15", "t16", "t17", "t19"):
            model = build_model(resolve(experiment))
            counts[experiment] = sum(item.numel() for item in model.parameters())
        self.assertEqual(counts["t14"], counts["t15"])
        self.assertEqual(counts["t15"], counts["t16"])
        self.assertEqual(counts["t15"], counts["t17"])
        self.assertEqual(counts["t15"], counts["t19"])
        self.assertGreater(counts["t15"], counts["t13"])


if __name__ == "__main__":
    unittest.main()
