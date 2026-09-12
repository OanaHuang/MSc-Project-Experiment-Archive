from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.spikepose.artifacts import verify_completed_runs


class RunCompletionTests(unittest.TestCase):
    def test_shared_e_s_gate_accepts_completed_run_with_checkpoints(self) -> None:
        with TemporaryDirectory() as temporary:
            run = Path(temporary) / "seed_42"
            (run / "checkpoints").mkdir(parents=True)
            (run / "status.json").write_text(
                json.dumps({"status": "completed"}), encoding="utf-8",
            )
            for checkpoint in ("last.pt", "best.pt"):
                (run / "checkpoints" / checkpoint).touch()
            verify_completed_runs((("model", run),), context="20-epoch verification")

    def test_shared_e_s_gate_rejects_incomplete_run(self) -> None:
        with TemporaryDirectory() as temporary:
            run = Path(temporary) / "seed_42"
            run.mkdir(parents=True)
            (run / "status.json").write_text(
                json.dumps({"status": "running"}), encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "status=running"):
                verify_completed_runs((("model", run),))


if __name__ == "__main__":
    unittest.main()
