# NTU M Motion-Trend Series

The M series separates real video time from ILIF simulation time.  Every real
frame is encoded independently with the shared SpikePose encoder before the
four frame-level heatmaps are fused.

| ID | History use | Fusion | Controlled question |
|---|---|---|---|
| M0 | unused | none | current-frame T0 reference |
| M1 | four independent frame predictions | uniform mean | does unmodelled history help? |
| M2 | first-three-frame constant-acceleration forecast | fixed 0.25 residual | is motion trend useful? |
| M3 | same causal forecast | learned per-joint current-evidence gate | is adaptive fusion better? |

M2 and M3 return the history-only forecast and current-only prediction as
intermediates during training.  Auxiliary heatmap supervision prevents the
final fusion loss from hiding a failed forecast branch.

Screen runs use seed 42 for 20 epochs. Main runs use seeds 42, 2071461405, and
365198782 for 120 epochs.
