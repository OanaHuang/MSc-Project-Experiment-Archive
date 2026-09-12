# Remaining spatial baselines

All jobs use the existing CLI, data loaders, heatmap loss, checkpoint/resume
format and output layout. Launch seed 42 only. MPII uses 140 epochs; the two
NTU baselines use the existing 20-epoch, 16-frame frame-reset source protocol
(5 epochs of head adaptation, then 15 of spatial fine-tuning).

| Experiment | Initialization | Output root |
|---|---|---|
| mpii_official_spikeyolo | scratch | Outputs_Thesis |
| mpii_official_spikformer | scratch | Outputs_Thesis |
| pilot20_ntu_spikepose_ann | mpii_official_spikepose_ann / best.pt | Outputs_Thesis_Pilot20 |
| pilot20_ntu_spikeyolo | mpii_official_spikeyolo / best.pt | Outputs_Thesis_Pilot20 |
| pilot20_ntu_spikformer | mpii_official_spikformer / best.pt | Outputs_Thesis_Pilot20 |

`initialization.source_output_root` explicitly resolves a formal MPII source
from a pilot NTU run. Existing configurations without this field retain their
previous source lookup behavior. `source_run_type: formal` explicitly permits
the formal MPII-to-pilot NTU transfer without relaxing checks for other jobs.
No checkpoint copying is needed. Both NTU baselines retain the existing eight
clips per batch, 16 frames per clip, and one optimizer step per batch. A real
full-finetuning batch was verified on the training GPU before queueing.

## SpikeYOLO adapter

The baseline is **SpikeYOLO-s P3 + the shared LinearHeatmapHead**, not the
unmodified detection network. `models/baselines/spikeyolo.py` implements the
scale-s YAML graph at BICLab/SpikeYOLO commit
`dc51f35d0b6e970cd157080c98157a41af4392f5`. The retained layer bodies in
`_spikeyolo_layers.py` are copied unchanged, including the T=1, D=4 ILIF
surrogate gradient. The accompanying AGPL-3.0 license is retained.

Feature layer 19 produces 128-channel stride-8 P3 features. The common linear
head projects to 16 joints and resizes to 64x64 heatmaps. Detection-only graph
descendants 20--26 are excluded because they do not contribute to that feature;
the previous placeholder `keep_layers: [0, 25]` has been corrected accordingly.
The adapted model has 16,738,512 parameters, not the complete detector's 23M.
Every retained parameter receives a gradient. No detector checkpoint is loaded.
The old external `weights/best.pt` has not been authenticated and is unused.

Only PyTorch is needed for these vendored layers: no dependency on an installed
Ultralytics detector package, its global settings, downloads or custom trainer.
Other model families use the same builder branches as before.

## Spikformer-T1 adapter

`models/baselines/spikformer.py` adapts the original MIT-licensed
Spikformer-8-384 image-classification backbone to the shared 16-joint linear
heatmap head.  The classifier and global spatial pooling are removed; the
stride-16 SPS/SSA feature map is projected to 16 heatmaps and resized to 64x64.
The two LayerNorm modules constructed but never called by the upstream block
are not retained, so every reported parameter participates in the pose graph.

Both MPII and NTU use one SNN simulation update per physical frame.  NTU keeps
the same 16-frame frame-reset protocol as SpikeYOLO, so video time is not
confused with the backbone simulation step.  The MPII model is trained from
scratch and its complete backbone/head checkpoint initializes the corresponding
NTU run through the existing strict `spatial_weights_only` loader.

## Evaluation

After MPII training, run `evaluate --experiment mpii_official_spikeyolo --seed 42`
to use official head-rectangle PCKh, joint exclusions and quarter-pixel decoding.
For NTU, training validation is not the paper's test result: evaluate each best
checkpoint using `tools/evaluate_fullvideo_subset.py` and the existing frozen
`fullvideo_xsub35` metadata (4,563 videos / 419,968 frames). Do not select
checkpoints on test scores. Recompute adapted-model costs before reporting them;
do not use detector paper FLOPs or the old 23M label as measured pose costs.
