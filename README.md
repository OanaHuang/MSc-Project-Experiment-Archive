# SpikePose: MSc Project Experiment Archive

This repository accompanies the dissertation **SpikePose: Lightweight Video Human Pose Estimation**. It preserves the experiment framework, model configurations, evaluation tools, and supporting documentation used to study lightweight spiking human-pose estimation on images and videos.

The maintained implementation lives in [`scripts/`](scripts/). For the complete command reference and experiment protocols, see the [experiment framework guide](scripts/README.md).

## Project Overview

SpikePose is a lightweight video human-pose estimator built around two components:

- A frame-reset spiking encoder that processes each RGB frame with one internal update.
- Motion-Aligned Heatmap Memory (MAM), which aligns historical joint heatmaps and combines them with the current prediction.

The project first studies spatial architecture design on MPII, including network depth, multi-scale fusion, neuron type, prediction heads, attention placement, and heatmap decoding. It then evaluates temporal processing on NTU RGB+D by comparing internal spiking updates, physical video frames, coordinate filters, memory variants, ANN baselines, and spiking baselines.

## Datasets

| Dataset | Role | Evaluation data |
| --- | --- | --- |
| MPII Human Pose | Spatial architecture and image-pose evaluation | 2,958 validation persons in the final comparison |
| NTU RGB+D 60 | Video-pose and temporal evaluation | 4,563 videos and 419,968 frames in the final subset |

NTU follows the Cross-Subject protocol. RGB images are used as model inputs, while the supplied skeleton data provides the joint annotations. Dataset files are not included in this repository and must be prepared separately for local training or evaluation.

## Main Results

### MPII spatial evaluation

| Model | PCKh@0.5 (%) | GFLOPs | Parameters (M) |
| --- | ---: | ---: | ---: |
| SpikePose-ANN | 87.05 | 8.44 | 7.518 |
| SpikePose, one update | 85.84 | 8.46 | 7.518 |
| SpikePose, two updates | 86.16 | 16.92 | 7.518 |

The second internal update improved PCKh by 0.32 points but nearly doubled the spatial computation. The single-update model was therefore retained for the video system.

### NTU video evaluation

| Model | PCKHB@0.5 (%) | MPJVE | MPJAccE | GFLOPs | Parameters (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| SpikePose without MAM | 83.63 | 4.87 | 8.02 | 8.459 | 7.520 |
| SpikePose with MAM | 83.81 | 3.50 | 5.50 | 8.462 | 7.523 |
| SpikePose-ANN with MAM | 84.93 | 3.03 | 4.69 | 8.445 | 7.523 |

Adding MAM reduced velocity error by **28.13%** and acceleration error by **31.46%**, while PCKHB increased from 83.63% to 83.81%. MAM added approximately 0.003M parameters and 0.0023 GFLOPs per frame.

Under the reported arithmetic-energy profile, SpikePose used 1.665 mJ per frame compared with 19.310 mJ for the matched ANN, a reduction of 91.4%. These values are analytical estimates from a separate profiling setting rather than measured hardware energy.

## Repository Layout

```text
scripts/
├── spikepose/             # Shared models, training, analysis and visualization
├── experiments/          # Canonical model configurations
├── ablations/            # Shared ablation groups and the B0-to-E9 route
├── MPII/                 # MPII data pipeline, training and evaluation
├── NTU_RGBD/             # NTU RGB+D temporal pipeline and experiment series
├── tools/                # Launchers, schedulers, audits and result summaries
├── tests/                # Regression and protocol tests
├── thesis_experiment/    # Retained supporting and publication experiment code
├── experiment_integrity_test.py
├── smoke_test.py
└── requirements.txt
```

The shared `spikepose` package is separated from dataset-specific data, metrics, and visualization code. MPII-specific configurations live under `scripts/MPII`, while real-frame temporal studies live under `scripts/NTU_RGBD`.

## Getting Started

Clone the repository and install the core dependencies:

```bash
git clone https://github.com/OanaHuang/MSc-Project-Experiment-Archive.git
cd MSc-Project-Experiment-Archive
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r scripts/requirements.txt
```

Before scheduling a full experiment, verify configuration compatibility and run a model forward-pass smoke test:

```bash
python3 scripts/experiment_integrity_test.py
python3 scripts/smoke_test.py
```

## Running Experiments

Train the MPII or NTU RGB+D baseline:

```bash
python3 scripts/MPII/train.py --experiment baseline
python3 scripts/NTU_RGBD/train.py --experiment baseline
```

Run a multi-GPU ablation batch:

```bash
python3 scripts/tools/run_ablation.py \
  --dataset mpii \
  --group structure \
  --batch-name mpii_140ep \
  --seeds 42 43 44 \
  --devices 0 1 2 3
```

Use `--resume --epochs <total>` to continue a compatible run from its last checkpoint. Without `--resume`, training starts from a new initialization.

## Outputs and Reproducibility

Experiment artifacts follow a canonical layout:

```text
Outputs_New/{dataset}/{category}/{batch}/{experiment}/seed_{seed}/
```

Each completed run can include the resolved configuration, checkpoints, training history, metrics, energy analysis, status metadata, and report visualizations. Shared comparisons reference existing runs rather than duplicating checkpoints.

The framework records analytical operation counts, firing rates, and arithmetic-energy estimates. Hardware energy measurements are stored separately and should be collected on an otherwise idle GPU. Analytical estimates and measured device energy are not interchangeable.

## Experiment Guides

- [Framework, training, outputs, and ablations](scripts/README.md)
- [MPII official baselines](scripts/MPII/OFFICIAL_BASELINES.md)
- [NTU contiguous protocol](scripts/NTU_RGBD/CONTIGUOUS_NTU60_PROTOCOL.md)
- [NTU cross-frame state experiments](scripts/NTU_RGBD/CROSS_FRAME_STATE_EXPERIMENTS.md)
- [NTU video-HPE solution series](scripts/NTU_RGBD/VIDEO_HPE_SOLUTION_SERIES.md)
- [Thesis and publication experiments](scripts/thesis_experiment/README.md)
- [Model naming conventions](scripts/thesis_experiment/MODEL_NAMING.md)
- [ICASSP 2027 workflow](scripts/thesis_experiment/ICASSP2027_WORKFLOW.md)

## Notes

- Large datasets, checkpoints, and generated experiment outputs are intentionally not tracked in this source repository.
- Legacy experiment identifiers remain supported so that existing runs can be resumed and audited against their saved resolved configurations.
- Reported energy figures should always be interpreted together with the corresponding measurement protocol.
