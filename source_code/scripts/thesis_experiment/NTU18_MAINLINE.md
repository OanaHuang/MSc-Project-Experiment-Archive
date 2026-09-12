# NTU-18 mainline migration gate

NTU-18 is the intended main joint layout for the thesis experiments. This file
is deliberately a gate rather than an invented mapping: the repository does not
currently contain an authoritative 18-joint definition.

Before creating or running NTU-18 experiment configs, freeze all of the
following in one versioned mapping module and protocol:

1. the 18 retained Kinect V2 joint indices and their output order;
2. left/right flip pairs and any centre joints;
3. the PCK-HB normalizing bone or another explicitly named NTU normalizer;
4. conversion rules for MPII-16 warm-start heads, including initialization of
   the two additional output channels;
5. visibility handling for missing or untracked NTU joints;
6. visualization edges and metric labels;
7. one equivalence test proving the raw NTU-25 cache is subsetted identically in
   frame-wise and temporal datasets.

The migration is complete only when the paper studies point to NTU-18
protocols, every NTU model reports `num_joints: 18`, and repository validation
rejects accidental 16- or 25-joint mainline configs.

The native NTU-25 Pilot8 study is outside this migration and remains supported
as an auxiliary comparison.
