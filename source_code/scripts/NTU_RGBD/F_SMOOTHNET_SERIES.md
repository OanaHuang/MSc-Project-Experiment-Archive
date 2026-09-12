# NTU F-SmoothNet Series

FS1-FS4 are coordinate-level SmoothNet post-processors trained on frozen F1-F4
predictions. They do not modify the SpikePose checkpoints.

| ID | Frozen source | Window | Primary accuracy | Temporal metrics |
|---|---|---:|---|---|
| FS1 | F1 | 32 | PCKhn@0.5 | NAccE, Accel, NAccel |
| FS2 | F2 | 32 | PCKhn@0.5 | NAccE, Accel, NAccel |
| FS3 | F3 | 32 | PCKhn@0.5 | NAccE, Accel, NAccel |
| FS4 | F4 | 32 | PCKhn@0.5 | NAccE, Accel, NAccel |

Frozen F models export separate contiguous train and validation coordinate
sequences. SmoothNet is fitted only on train predictions; reported comparisons
use validation sequences. Temporal differences never cross video boundaries.

```bash
.venv/bin/python scripts/NTU_RGBD/train_f_smoothnet_queue.py --dry-run
.venv/bin/python scripts/NTU_RGBD/train_f_smoothnet_queue.py --launch
```
