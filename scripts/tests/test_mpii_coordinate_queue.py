from argparse import Namespace
from pathlib import Path

from scripts.tools.run_coordinate_representation import command


def test_coordinate_queue_is_fixed_to_twenty_epochs_and_seed_42():
    args = Namespace(
        batch_name="profile_b_ep020_seed42", output_root=Path("Outputs_New"),
    )
    cc = command("coordinate_classification", [0, 1], args)
    rg = command("coordinate_regression", [2, 3], args)
    for value, group in ((cc, "coordinate_classification"),
                         (rg, "coordinate_regression")):
        assert value[value.index("--group") + 1] == group
        assert value[value.index("--epochs") + 1] == "20"
        assert value[value.index("--seeds") + 1] == "42"
