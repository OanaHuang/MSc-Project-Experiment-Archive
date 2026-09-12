from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.tools.run_ablation import write_random_seed_plan


class RandomSeedPlanTests(unittest.TestCase):
    def test_plan_records_random_seeds_and_targets(self) -> None:
        with TemporaryDirectory() as temporary:
            batch_dir = Path(temporary)
            args = Namespace(
                dataset="mpii", category="main_route",
                batch_name="mpii_main_route_20ep",
            )
            targets = [batch_dir / "e0/seed_101", batch_dir / "e0/seed_202"]
            path = write_random_seed_plan(
                batch_dir, args, ["e0"], [101, 202], targets,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "system_secure_random")
            self.assertEqual(payload["seeds"], [101, 202])
            self.assertEqual(payload["experiments"], ["e0"])
            self.assertEqual(payload["targets"], [str(path) for path in targets])


if __name__ == "__main__":
    unittest.main()
