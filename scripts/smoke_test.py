from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.spikepose.experiments import (
    experiment_roots, resolve_config, validate_config,
)
from scripts.spikepose.models import build_model


def main() -> None:
    root = PROJECT_ROOT
    experiments = (
        "baseline", "head1x", "stage123_clean", "stage1", "stage12",
        "stage123", "add", "lif", "relu",
        "spike_head", "t1", "t4", "four_stage", "mem_ann",
        "mem_spikefpn_annhead", "mem_fpn", "multiconv_fpn", "sew_fpn",
        "resformer_fpn", "resformer_s3", "resformer_s4",
        "s0", "s1", "s2", "s3", "s4", "s5", "s6",
        "cc1", "cc2", "cc3", "rg1", "rg2",
    )
    for name in experiments:
        config = resolve_config(
            experiment_roots(root, "mpii"), name,
            root / "scripts" / "MPII" / "configs" / "task.yaml",
            root / "scripts" / "MPII" / "configs" / "training.yaml",
        )
        validate_config(config)
        model = build_model(config).eval()
        with torch.no_grad():
            output = model(torch.randn(1, 3, 256, 256))
        if torch.is_tensor(output):
            assert output.shape == (1, 16, 64, 64), (name, output.shape)
            output_shape = tuple(output.shape)
        else:
            assert output["output_type"] in {
                "coordinate_classification", "coordinate_regression",
            }
            output_shape = {
                key: tuple(value.shape) for key, value in output.items()
                if torch.is_tensor(value)
            }
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"{config['name']} ({config.get('legacy_reference', 'n/a')}): "
            f"{output_shape}, {parameters:,} parameters"
        )


if __name__ == "__main__":
    main()
