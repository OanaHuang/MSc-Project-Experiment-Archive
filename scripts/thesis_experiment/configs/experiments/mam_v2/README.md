# MAM-v2 paper configurations

Only configurations used by the frozen paper roster or the retained NTU-25
warm start remain here:

- `mamv2_fullcs_p00_source` and `mamv2_fullcs20` form the full-CS Pilot path;
- `mamv2_p00_source`, `mamv2_p06_noalign` and `mamv2_p07_fixed_decay` preserve
  the two selected S010 ablations and their seed-matched source;
- `pilot20_mamv2_noresidual` is the full-CS residual-fusion ablation.
- `pilot20_mamv2_nomotiontoken`, `pilot20_mamv2_nooffsethead`, and
  `pilot20_mamv2_nomemoryupdate` complete the Table 5(a) structural ablation.

The no-motion-token cell keeps coarse coordinate alignment, but replaces the
17-D learned token with a neutral tensor and disables the residual offset head.
The dynamic leak module remains enabled and can learn its neutral-input bias.
The no-memory-update cell still forms and fuses the aligned one-step candidate,
but writes the raw current heatmap into recurrent memory for the next frame.

The exploratory P01-P05/P08-P12 and Pilot40 matrices were removed from this
branch. Their history remains available on `main` and in earlier commits.

## KTP prior pilot

The separate `mam_ktp_pilot20` study is a causal 2x2 ablation over the Core
MAM motion tokens:

- `mamv2_fullcs20`: KPA off, TPA off (the retained Core MAM checkpoint);
- `pilot20_mamv2_kpa`: KPA on, TPA off;
- `pilot20_mamv2_tpa`: KPA off, causal TPA on;
- `pilot20_mamv2_ktp`: KPA on, causal TPA on.

KPA uses the MPII-16 skeleton topology plus a learnable global joint affinity.
TPA uses consecutive-frame topology plus a learnable lower-triangular global
temporal affinity. The lower-triangular mask and token-wise LayerNorm preserve
the online MAM contract: frame `t` never consumes features from a later frame.
These are KTPFormer-inspired prior adapters on MAM's 17-D motion tokens, not a
claim that the original 3D KTPFormer architecture is reproduced unchanged.
