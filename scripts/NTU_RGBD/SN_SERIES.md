# NTU SN Series

The SN series is a training-free temporal post-processing comparison using the
frozen T0 original-video coordinate predictions. All runs use seed 42.

| ID | Method | Parameters |
|---|---|---|
| SN0 | raw T0 reference | identity |
| SN1 | Savitzky-Golay | window 7, polynomial 2 |
| SN2 | One Euro Filter | 30 FPS, min cutoff 1.0, beta 0.007, derivative cutoff 1.0 |

Formal outputs live under:

```text
Outputs_New/ntu_rgbd/sn_series/training_free_v1/
  sn0_raw/seed_42/
  sn1_savgol_w7_p2/seed_42/
  sn2_one_euro/seed_42/
```

Each experiment reports PCKhn@0.5, NAccE, Accel and NAccel. Videos use the
same T0 rendering contract: five unique sample IDs drawn without replacement
from sorted valid S010 validation IDs using seed 42, at most 120 frames, 15 FPS,
native resolution and MP4V encoding.
