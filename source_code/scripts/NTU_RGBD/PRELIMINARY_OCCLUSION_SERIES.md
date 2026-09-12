# NTU Preliminary Occlusion Validation Series

All experiments consume cached, frozen T0 coordinate predictions. Neural
refiners use a causal 16-frame window and synthetic masks applied only to
originally tracked joints. The unmasked ground truth remains available for
repair supervision.

| ID | Method | Train | Purpose |
|---|---|---:|---|
| PV0 | T0 identity | no | clean and masked observation baseline |
| PV1 | T0 masked identity | no | quantify degradation under masking |
| PV2 | last reliable observation | no | minimal causal baseline |
| PV3 | confidence Kalman | no | traditional causal filter baseline |
| PV4 | two-layer causal GRU | yes | parameter-matched ANN baseline |
| PV5 | coordinate ILIF | yes | basic SNN baseline |
| PV6 | confidence-dynamic ILIF | yes | confidence-controlled memory |
| PV7 | confidence-dynamic graph ILIF | yes | full preliminary model |

PV0 evaluates clean cached T0 predictions. PV1-PV7 use the same deterministic
validation masks; training masks change deterministically with each epoch.
