from __future__ import annotations

import unittest

import numpy as np

from scripts.MPII.evaluation.heatmap_ablation import (
    HeatmapVariant, flip_back_heatmaps,
)


class HeatmapDecodingTests(unittest.TestCase):
    def test_variant_rejects_shift_without_flip(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires flip_test"):
            HeatmapVariant.from_dict({
                "id": "invalid", "decoder": "argmax", "flip_shift": True,
            })

    def test_flip_back_swaps_mpii_left_and_right_joints(self) -> None:
        heatmaps = np.zeros((1, 16, 2, 4), dtype=np.float32)
        heatmaps[0, 0, 0, 0] = 1.0
        restored = flip_back_heatmaps(heatmaps, shift=False)
        self.assertEqual(restored[0, 5, 0, 3], 1.0)
        self.assertEqual(restored[0, 0, 0, 0], 0.0)

    def test_flip_shift_matches_existing_evaluation_protocol(self) -> None:
        heatmaps = np.zeros((1, 16, 1, 4), dtype=np.float32)
        heatmaps[0, 6, 0, 1] = 1.0
        restored = flip_back_heatmaps(heatmaps, shift=True)
        # Horizontal restore moves x=1 to x=2, then the protocol shifts right.
        self.assertEqual(restored[0, 6, 0, 3], 1.0)

    def test_official_compatible_variant(self) -> None:
        variant = HeatmapVariant.from_dict({
            "id": "flip_quarter",
            "decoder": "quarter",
            "flip_test": True,
            "flip_shift": True,
        })
        self.assertTrue(variant.flip_test)
        self.assertTrue(variant.flip_shift)
        self.assertEqual(variant.decoder, "quarter")
        self.assertFalse(variant.udp)


if __name__ == "__main__":
    unittest.main()
