# Prometheus pitch video

Narration uses local synthetic speech. Simulation footage is explicitly labeled.

Prometheus is a robot learning and inspection workbench for the Intel Physical AI challenge. These visuals show our MuJoCo simulation.

The goal is to pick blocks from clutter and place them correctly. Our learning campaign focuses on blue plate placement. A separate application adds quality inspection.

An overhead camera shows the target and destination. A wrist camera provides local grasp geometry. Matching these views is essential before transferring simulation data.

We compare ordinary images with a causal X cue. A selected target is tracked only while confidence holds. Whether this improves learned behavior remains an experiment.

We prepared sixty four demonstrations and a thirty two experiment matrix across ACT, Smol VLA, target cues, augmentation and data sources. Real data training launched. Mixed simulation runs remain gated.

This is a recorded successful expert rollout in MuJoCo. The arm grasps, transports and releases the block onto the blue plate. This is simulation, not a real arm or learned policy success claim.

Physical AI Studio connects collection and policy deployment. We implemented OpenVINO inference checks, Intel station deployment tools, and an anomaly inspection application.

Real policy driven motion has been tested. Reliable physical pick and place is still unproven. We added output checks after reproducing out of range policy targets.

Prometheus makes the work inspectable through source, data contracts, simulation and test evidence. The next milestone is comparable real world trials across the same evaluation scenes.
