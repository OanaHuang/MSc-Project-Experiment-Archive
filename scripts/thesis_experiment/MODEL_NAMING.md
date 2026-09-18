# Paper Model Identity and Soft-Coordinate Scale Diagnostic

Audit date: 11 September 2026. The naming follows the method section and Tables 1, 4, and 5 in the corresponding thesis draft.

## Reproducibility scope and execution boundary

This document provides more than a name mapping. Future work must first confirm that the paper method, experiment configuration, code version, checkpoint, evaluation protocol, and reported value belong to the same traceable chain. Only then should it analyse whether the softmax-derived soft coordinates cause a motion-estimation scale mismatch.

This document records model identity and diagnostic scope. Updating this document does not start training, change inference algorithms, or alter reported results.

### Diagnostic sequence

1. **Confirm the working directory.** Use `scripts/thesis_experiment` under the project root. Record `pwd`, Git HEAD, working-tree changes, and the Python environment. Do not substitute model-family descriptions from the repository README, NTU-18 plans, the NTU-25 preview, or another worktree for the current V2.6 paper experiments using 16 joints.
2. **Verify model identity.** Read `evaluation_provenance.json`, the resolved training configuration, lineage information, and checkpoint for each formal evaluation listed below. For SpikePose, verify the SHA-256, epoch 13, and seed 42 stated here. Read the provenance for each baseline and ablation independently; do not reuse the main model's epoch. Confirm the 16-joint layout, 64 x 64 heatmaps, sigma 2, one SNN update per frame, MAM settings, and initialisation source.
3. **Confirm the implementation.** Distinguish current code, training-time code, and the code used for the 9 September 2026 evaluation. The evaluation provenance records uncommitted changes, so its Git commit cannot reconstruct the complete source. Preserve relevant files and hashes. Inspect the builder path from spatial heatmaps to MAM, MAM softmax, displacement residuals, warping, state continuation, and DARK decoding. If historical code cannot be reconstructed, state that limitation explicitly.
4. **Fix the evaluation protocol.** The current NTU60 paper protocol uses 4,563 complete videos, 419,968 frames, the MPII-style 16-joint layout, and one fixed tube crop per full video. The SNN resets on every frame. MAM state continues across inference chunks and resets only at source-video boundaries. Exclude older per-window reset, per-window crop, sampled-2x16, and NTU-25 results. For small diagnostics, save a fixed validation-video list and keep coordinate and crop definitions unchanged. Do not present small-sample results as the complete paper evaluation.
5. **Run read-only inference diagnostics first.** Use the verified checkpoint. Record the range, peak, and background quantiles of raw heatmaps before MAM; softmax maximum probability and entropy; soft-coordinate and DARK errors against ground truth; coarse, residual, and final displacement errors against ground-truth displacement; residual magnitude; and saturation rate. Group by true motion magnitude, use only valid joints in adjacent frames, compare all quantities in heatmap coordinates, and record the valid sample count. Include a zero-displacement prediction as a reference.
6. **Label same-checkpoint comparisons as diagnostics.** Store the original implementation, alignment disabled at inference, ground-truth displacement oracle, and temperature variants separately. Use ground truth only for an explicitly labelled oracle diagnostic. Select temperature only on the validation set. Changing soft-coordinate extraction changes the offset and gate input distribution, so an unadapted inference variant cannot be presented as a formally repaired model result.
7. **Let evidence determine the repair.** Report findings as confirmed, unconfirmed, and next steps. If real heatmaps show substantial displacement compression, propose a coordinate-extraction correction, training adaptation, and re-evaluation of affected ablations. Do not claim a repair from a formula change alone, and do not invalidate all PCK results solely from a synthetic Gaussian counterexample.

Each diagnostic output must include the command and environment record, model/code/data inventory with hashes, grouped metric CSV files, displacement comparison figures, and a Markdown conclusion. Write these files to a new diagnostic directory and preserve existing checkpoints and evaluations.

### Completed and outstanding work

- Completed: read the current Overleaf method definitions and main tables; align the main-model and baseline metrics; verify the local main-model checkpoint hash; inspect the softmax and residual code in two local projects; and reproduce the synthetic Gaussian example.
- Outstanding: verify the current source against historical run code; diagnose raw heatmaps and motion estimates from the real checkpoint; and perform any repair, training, or formal re-evaluation.
- Synthetic example: for a 64 x 64 heatmap with sigma 2 and a peak moving from (16, 24) to (20, 24), the full Gaussian produces a coarse displacement of 0.0320867 pixels and the Gaussian truncated at 3 sigma produces 0.0320372 pixels. A residual limited to 2 pixels per axis still cannot recover the 4-pixel horizontal displacement. This result demonstrates a mismatch only at this input scale.
- The historical epoch-20 log reports a mean sampled residual of approximately 0.0447 pixels and a saturation rate of zero. It is not a full-validation statistic for the best checkpoint at epoch 13 and does not establish that the coarse displacement is correct.

The paper defines **SpikePose** as the complete single-step SNN encoder and MAM system, and **SpikePose w/o MAM** as the framewise SNN variant. Diagnostic reports, figures, and captions should use the display names below. Experiment IDs and checkpoint paths remain provenance identifiers.

## NTU60 main model and baselines

| Paper display name | Experiment ID | Previous report or internal alias |
| --- | --- | --- |
| SpikePose | `mamv2_fullcs20` | Core MAM, MAM, mam |
| SpikePose w/o MAM | `mamv2_fullcs_p00_source` | SpikePose Frame, Frame baseline, p00 |
| SpikePose-ANN w/o MAM | `pilot20_ntu_spikepose_ann` | SpikePose-ANN, spikepose_ann |
| SpikePose-ANN + MAM | `pilot20_ntu_spikepose_ann_mam` | ANN + MAM, spikepose_ann_mam |
| SpikeYOLO-Frame | `pilot20_ntu_spikeyolo` | SpikeYOLO, spikeyolo |
| ResNet-50 | `pilot20_ntu_simplebaseline_r50` | SimpleBaseline ResNet-50, simplebaseline_r50 |

`(Ours)` identifies table ownership and is not a model name. Used alone, MAM means the motion-aligned heatmap memory module. `MAM V2` and `mam_v2` identify an implementation version.

The mapping between the main model and its framewise baseline is verified by evaluation provenance under `Outputs_Thesis_Pilot20/reset_policy_validation/20260909_table4/`. Experiment mappings for the other Table 1 baselines are defined in the `MODELS` collection in `tools/run_fullvideo_evaluation_queue.py`.

### Main-model identity check

The main-model paper evaluation uses `Outputs_Thesis_Pilot20/runs/ntu60_cs/mam_v2_fullcs/mamv2_fullcs20/seed_42/checkpoints/best.pt`, at epoch 13.

The measured SHA-256 for the local `best.pt` at the same relative path is `4ea67827c0f7ee837e0f8544a98ade41466f234b465c42479ebba3aaeb908468`. It exactly matches the `checkpoint_sha256` stored by the corresponding paper evaluation.

`mam/video_reset/summary.json` reports PCK 83.8055987786%, MPJVE 3.4992037571, and MPJAccE 5.4979759527. Rounded values match the current paper result of 83.81 / 3.50 / 5.50. `matched_video_crop/p00/summary.json` reports 83.6309805483% / 4.8685491130 / 8.0213571577, matching the paper's framewise baseline result of 83.63 / 4.87 / 8.02.

These records confirm the checkpoint and evaluation-result identity. The provenance also records uncommitted code changes, so the stated Git commit alone cannot establish byte-for-byte identity between all current local files and the server source used for the evaluation. The MAM V2 files in the two local copies differ, but both apply an unscaled softmax directly to raw heatmaps and use a bounded tanh displacement residual.

## MAM ablations

Table 4 marks component variants with check marks. The following diagnostic names expand those component definitions; they are not presented as verbatim Table 4 row names.

| Diagnostic display name | Experiment ID | Actual meaning |
| --- | --- | --- |
| SpikePose w/o motion alignment | `mamv2_p06_noalign` | Disables historical-heatmap warping and displacement supervision. |
| SpikePose w/o adaptive gate | `mamv2_p07_fixed_decay` | Retains a learnable per-joint retention value that does not depend on the current input; it does not fix retention permanently at 0.08. |
| SpikePose w/o residual fusion | `pilot20_mamv2_noresidual` | Disables the direct residual-fusion path from the current heatmap while retaining the displacement-residual branch. |

The first two formal results use `variants/setups_full/seed_42`; do not select the default S010 development runs based only on their experiment IDs. The `evaluation_provenance.json` files in the corresponding `no_alignment`, `fixed_retention`, and `no_residual` directories provide the identifying evidence.

Keep these concepts separate:

- **Residual offset:** the residual in `coarse + residual` that corrects displacement, configured by `memory_use_residual_offset`.
- **Residual fusion:** the output-fusion path `H + gamma * (M - H)`, configured by `memory_use_current_direct_path`.

For `pilot20_mamv2_noresidual`, residual fusion is false and residual offset is true. The experiment therefore must not be described as removing residual displacement.

## Diagnostic result names and scope

- Use `SpikePose` and `SpikePose w/o MAM` for the original model results.
- When alignment is temporarily disabled at inference for the same checkpoint, use `SpikePose - alignment disabled at inference (diagnostic)`. Keep it distinct from the independently trained `SpikePose w/o motion alignment`.
- Label ground-truth alignment as `SpikePose - GT alignment (oracle diagnostic)`.
- Label temperature variants as `SpikePose - softmax temperature tau=... (diagnostic)`.
- `SP w/o M` in Table 5 is the defined abbreviation of `SpikePose w/o MAM`. Filter reports may expand it to `SpikePose w/o MAM + EMA / One Euro / SG`.
- KPA, TPA, and KTP extensions do not belong to the current Table 1 SpikePose model. Do not mix their checkpoints into the main-model diagnostic.
- Metrics from the same model under different crops, state-reset policies, data subsets, or evaluation versions require distinct protocol labels. A shared name does not make the results interchangeable.

## Remaining paper-name ambiguity

The method section defines SpikePose as the complete encoder-MAM system. MPII Table 2, however, uses `SpikePose (T=1)` and `SpikePose (T=2)` for the spatial encoder, while nearby text uses `SpikePose-SNN`. The `SpikePose-ANN` row in MPII Table 2 also refers to the spatial encoder and must not be read as including MAM.

A later paper revision should rename the MPII rows to `SpikePose w/o MAM (T_snn=1)` and `SpikePose w/o MAM (T_snn=2)`, rename the ANN row to `SpikePose-ANN w/o MAM`, and align the surrounding text. This document records the recommendation and does not change Overleaf.
