from __future__ import annotations

from pathlib import Path
import importlib.util
import unittest


# Load this dependency-free module directly. Importing the spikepose package
# intentionally initializes the PyTorch model stack, which is not needed here.
PATHS_FILE = Path(__file__).parents[1] / "spikepose" / "artifacts" / "paths.py"
SPEC = importlib.util.spec_from_file_location("output_paths", PATHS_FILE)
assert SPEC and SPEC.loader
PATHS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATHS)
canonical_output_parts = PATHS.canonical_output_parts
output_run_dir = PATHS.output_run_dir
validate_seed_targets = PATHS.validate_seed_targets
existing_seed_values = PATHS.existing_seed_values
generate_random_seeds = PATHS.generate_random_seeds


class OutputPathCompatibilityTests(unittest.TestCase):
    def test_old_main_route_ids_resolve_to_canonical_path(self) -> None:
        path = output_run_dir(
            Path("Outputs_New"), "mpii", "main_route",
            "mpii_main_route_20ep", "mem_spikefpn_annhead", 42,
        )
        self.assertEqual(
            path,
            Path("Outputs_New/mpii/architecture_evolution/profile_b/")
            / "e5_spike_fpn_ann_head/seed_42",
        )

    def test_old_backbone_ids_resolve_to_canonical_path(self) -> None:
        self.assertEqual(
            canonical_output_parts(
                "mpii", "backbone_fpn", "mpii_bf_20ep", "resformer_fpn",
            ),
            ("architecture_evolution", "profile_b", "e9_resformer_stages34"),
        )

    def test_training_schedule_keeps_model_id_but_renames_batch(self) -> None:
        self.assertEqual(
            canonical_output_parts(
                "mpii", "training_extension", "constant_1e6",
                "resformer_fpn",
            ),
            ("training_schedule", "constant_lr_1e-6_ep210", "resformer_fpn"),
        )

    def test_evolution_ids_resolve_to_named_iterations(self) -> None:
        self.assertEqual(
            canonical_output_parts(
                "mpii", "main_route", "evolution_profile_b", "e1",
            ),
            ("architecture_evolution", "profile_b", "e1_single_scale_head"),
        )

    def test_b0_resolves_to_shared_evolution_baseline(self) -> None:
        self.assertEqual(
            canonical_output_parts(
                "mpii", "pose_ablation", "mpii_pose_20ep", "baseline",
            ),
            ("architecture_evolution", "profile_b", "b0_original_baseline"),
        )

    def test_preliminary_runs_remain_separate(self) -> None:
        self.assertEqual(
            canonical_output_parts(
                "mpii", "main_route", "mpii_main_route_20ep", "head1x",
            ),
            ("architecture_search", "preliminary_v1_ep020", "single_head"),
        )

    def test_complete_main_route_launcher_resolves_all_eleven_runs(self) -> None:
        experiments = (
            "baseline", "e0", "e1", "e2", "four_stage", "mem_ann",
            "mem_spikefpn_annhead", "mem_fpn", "resformer_s3",
            "resformer_s4", "resformer_fpn",
        )
        resolved = {
            canonical_output_parts(
                "mpii", "main_route", "mpii_main_route_20ep", experiment,
            )
            for experiment in experiments
        }
        self.assertEqual(len(resolved), 11)
        self.assertTrue(all(parts[:2] == (
            "architecture_evolution", "profile_b",
        ) for parts in resolved))

    def test_unrelated_paths_are_unchanged(self) -> None:
        self.assertEqual(
            canonical_output_parts("mpii", "custom", "trial", "a/b"),
            ("custom", "trial", "a_b"),
        )

    def test_ntu_t_series_uses_simple_paths(self) -> None:
        self.assertEqual(
            output_run_dir(
                Path("Outputs_New"), "ntu_rgbd", "t_series",
                "clip4_256_20ep", "t5", 42,
            ),
            Path("Outputs_New/ntu_rgbd/t_series/clip4_256_20ep/t5/seed_42"),
        )

    def test_heatmap_ablation_has_an_independent_output_tree(self) -> None:
        self.assertEqual(
            canonical_output_parts(
                "mpii", "heatmap_decoding", "heatmap_v1", "b0",
            ),
            ("heatmap_decoding_ablation", "v1", "b0"),
        )
        self.assertEqual(
            output_run_dir(
                Path("Outputs_New"), "mpii", "heatmap_decoding",
                "heatmap_v1_smoke", "b0", 42,
            ),
            Path(
                "Outputs_New/mpii/heatmap_decoding_ablation/"
                "v1_smoke/b0/seed_42"
            ),
        )

    def test_training_profile_ablation_uses_named_p_series_paths(self) -> None:
        expected_names = (
            "p0_profile_a_baseline",
            "p1_small_normal_init",
            "p2_warmup_cosine_schedule",
            "p3_gradient_clipping",
            "p4_scale_augmentation",
            "p5_profile_b_full",
        )
        actual = tuple(
            canonical_output_parts(
                "mpii", "training_profile", "profile_ablation_v1_ep020",
                f"p{index}",
            )
            for index in range(6)
        )
        self.assertEqual(
            actual,
            tuple(
                ("training_profile_ablation", "profile_a_to_b", name)
                for name in expected_names
            ),
        )
        self.assertEqual(
            output_run_dir(
                Path("Outputs_New"), "mpii", "training_profile",
                "profile_ablation_v1_ep020", "p5", 42,
            ),
            Path(
                "Outputs_New/mpii/training_profile_ablation/profile_a_to_b/"
                "p5_profile_b_full/seed_42"
            ),
        )
    def test_stage_fusion_uses_profile_b_output_tree(self) -> None:
        expected = {
            "s0": "s0_stage4_only",
            "s1": "s1_stages34",
            "s2": "s2_stages234",
            "s3": "s3_stages1234",
            "s4": "s4_without_stage2",
            "s5": "s5_without_stage3",
            "s6": "s6_without_stage4",
        }
        for experiment, directory in expected.items():
            self.assertEqual(
                canonical_output_parts(
                    "mpii", "stage_fusion", "profile_b_seed42", experiment,
                ),
                ("stage_fusion", "profile_b", directory),
            )

    def test_duplicate_seeds_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate seeds"):
            validate_seed_targets(
                Path("Outputs_New"), "mpii", "main_route",
                "mpii_main_route_20ep", ("baseline",), [43, 43], False,
            )

    def test_existing_seed_is_not_overwritten(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = output_run_dir(
                root, "mpii", "main_route", "mpii_main_route_20ep",
                "baseline", 42,
            )
            target.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                validate_seed_targets(
                    root, "mpii", "main_route", "mpii_main_route_20ep",
                    ("baseline",), [42], False,
                )

    def test_new_seed_is_accepted(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            targets = validate_seed_targets(
                Path(temporary), "mpii", "main_route",
                "mpii_main_route_20ep", ("baseline", "e0"), [43, 44], False,
            )
            self.assertEqual(len(targets), 4)
            self.assertTrue(all(not target.exists() for target in targets))

    def test_resume_requires_last_checkpoint(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "Cannot resume"):
                validate_seed_targets(
                    Path(temporary), "mpii", "main_route",
                    "mpii_main_route_20ep", ("baseline",), [43], True,
                )

    def test_random_seeds_are_unique_and_exclude_recorded_values(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "experiment/seed_42").mkdir(parents=True)
            seeds = generate_random_seeds(root, 2)
            self.assertEqual(len(seeds), 2)
            self.assertEqual(len(set(seeds)), 2)
            self.assertNotIn(42, seeds)
            self.assertTrue(all(1 <= seed < 2**31 for seed in seeds))
            self.assertEqual(existing_seed_values(root), {42})

    def test_random_seed_count_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 1"):
            generate_random_seeds(Path("Outputs_New"), 0)


if __name__ == "__main__":
    unittest.main()
