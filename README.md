# SpikePose: MSc Project Experiment Archive

This repository accompanies the dissertation **SpikePose: Lightweight Video Human Pose Estimation**. It contains the experiment records, results, configurations, source-code snapshot, logs, and prediction outputs used in the project.

For a searchable summary of the experiments, open [msc_project_experiment_index.xlsx](msc_project_experiment_index.xlsx).

## Project Overview

SpikePose is a lightweight video human pose estimator built around two components:

- A frame-reset spiking encoder that processes each RGB frame with one internal update.
- Motion-Aligned Heatmap Memory (MAM), which aligns historical joint heatmaps and combines them with the current prediction.

The project first studies spatial architecture design on MPII, including network depth, multi-scale fusion, neuron type, prediction heads, attention placement, and heatmap decoding. It then evaluates temporal processing on NTU RGB+D by comparing internal spiking updates, physical video frames, coordinate filters, memory variants, ANN baselines, and spiking baselines.

## Datasets

| Dataset | Role | Evaluation data |
| --- | --- | --- |
| MPII Human Pose | Spatial architecture and image-pose evaluation | 2,958 validation persons in the final comparison |
| NTU RGB+D 60 | Video-pose and temporal evaluation | 4,563 videos and 419,968 frames in the final subset |

NTU follows the Cross-Subject protocol. RGB images are used as model inputs, while the supplied skeleton data provides the joint annotations.

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

## Code Architecture

The source snapshot is stored under:

```text
msc project data/source_code/scripts/
├── spikepose/
│   ├── models/
│   │   ├── neurons/       # I-LIF, LIF and activation layers
│   │   ├── blocks/        # Convolutional and residual blocks
│   │   ├── backbones/     # Spatial feature extractors
│   │   ├── necks/         # Multi-scale fusion and Spike-FPN
│   │   └── heads/         # Heatmap and coordinate heads
│   ├── training/          # Shared training components
│   ├── analysis/          # Complexity and energy analysis
│   └── visualization/     # Model and prediction visualisation
├── experiments/           # Shared experiment configurations
├── ablations/             # Ablation groups and the E0-E9 route
├── MPII/
│   ├── datasets/
│   ├── evaluation/
│   ├── experiments/
│   ├── train.py
│   └── evaluate.py
├── NTU_RGBD/
│   ├── datasets/
│   ├── evaluation/
│   ├── experiments/
│   ├── train.py
│   └── evaluate.py
├── tools/                 # Experiment launchers and result summaries
├── tests/                 # Integrity and smoke tests
└── thesis_experiment/     # Retained supporting experiment code
