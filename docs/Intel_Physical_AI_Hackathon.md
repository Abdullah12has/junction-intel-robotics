# Intel Physical AI Hackathon

Source: [Intel_Physical_AI_Hackathon.pdf](Intel_Physical_AI_Hackathon.pdf). Page-by-page text extraction; diagrams and slide layout remain in the original PDF.

## Page 1

```text
Stop Simulating, Start Moving:
See It. Detect It. Act on It. The
Intel Physical AI Challenge
```

## Page 2

```text
Intel x Ignite Next Hackathon Support Team




  Albert van Houten   Daniil Lyakhov   Jayabalaji Sathiyamoorthi   Olinca Beltran




  Rajesh Gangireddy   Ada Betancourt         Marvin Mueller        Ricky Watts
                                                                           @2026 Intel Corporation   2
```

## Page 3

```text
Intel’s Challenge Overview
Build an end-to-end Physical AI solution for robotic defect detection and intelligent manipulation using the Intel
Physical AI Studio, Anomalib, and OpenVINO software stack.
Participants will design a robotics workflow that combines Vision-Language-Action (VLA) capabilities through
Intel Physical AI Studio with Anomalib-based visual anomaly and defect detection.
The solution should enable the robot to perceive objects, identify defects, reason about the required action,
and execute the appropriate manipulation task.
The completed solution will then be optimized and deployed on Intel® Core Ultra Series 3 using OpenVINO,
leveraging available CPU, GPU, and NPU acceleration to improve inference performance, latency, and
deployment efficiency.
Primary task: distinguish acceptable (“perfect”) from anomalous (“defective”) objects, then physically sort or
otherwise respond according to the rules revealed at kickoff.
Closed-loop sequence: PERCEIVE → DETECT → REASON → ACT → OPTIMIZE and demonstrate the entire
pipeline.



                                                                                                         @2026 Intel Corporation   3
```

## Page 4

```text
Required Technology Stack




                            @2026 Intel Corporation   4
```

## Page 5

```text
End-to-End Technical Workflow




                                @2026 Intel Corporation   5
```

## Page 6

```text
Technical Implementation Expectations




                                        @2026 Intel Corporation   6
```

## Page 7

```text
Judging Criteria




                   @2026 Intel Corporation   7
```

## Page 8

```text
HW/SW at-a-Glance
Hardware:
• Compute: one of either of the below
  – HP OmniBook Ultra (C2NM8AV) Intel® Core Ultra X9 Series 3 (388H) AI PC
  – Dell XPS 16 [DA16260] Intel® Core Ultra X9 Series 3 (388H) AI PC
• Arm: LeRobot SO-ARM101 robotic arm; Assembled 1 Leader + 1 Follower + 1 wrist
  camera
• Cameras: one (1) RealSense Depth Camera 415 and one (1) Pro QHD 2K/60fps
  Webcam
• Add’l equipment: camera desk mounts, USB Hub, Legos/bowls, robot safety
  printed guidelines

Software:

• Needs to be installed as part of the challenge
  – Physical AI Studio
  – AnomaLib v2.7.0

• Pre-Installed SW: Ubuntu v26.04, NPU Driver, iGPU Driver

• Training: GPU Cloud Instance provided via Modal                                 @2026 Intel Corporation   8
```

## Page 9

```text
End Goal                                                                                                     Defective objects




                                     ACT

                                                           Anomalib
                                                                                          SmolVLA

What Teams Must Deliver
• Working Physical AI prototype: closed-loop SO-101 solution from perception to action.                        Perfect objects
• VLA + Anomalib integration: VLA robotics workflow with visual anomaly / defect detection.
• OpenVINO-optimized deployment: optimized inference pipeline on Intel Core Ultra Series 3.
• Live demonstration: execute the Challenge-Day task and explain the architecture and optimization choices

                                                                                                                        @2026 Intel Corporation   9
```

## Page 10

```text
   Happy
   Hacking
https://developer.robotics.intel.com




                                       @2026 Intel Corporation   10
```

## Page 11

```text
End-to-End Physical AI Workflows
A universal runtime that connects models, backends, and robots to standardize and scale deployment




 Physical AI Studio                                                                     Robotics AI Suite                                                                               OpenVINO Physical AI Framework

Fine-tune imitation learning                                                         Standardize workflows                                                                              Deploy VLA policies to production
• Imitation Learning: teach robots to                                                • Plug-and-play integration: calibrate data                                                        • Inference to production: in a few lines of Python
  perform tasks through from human                                                     from diverse inputs with integrations                                                            • Heterogeneous compute: optimized for Intel®
  demonstrations                                                                       across diverse hardware set-ups                                                                    Core and Intel® Core Ultra processors with
• State-of-the-Art Policies: run ACT,                                                • ROS 2 Ready: optimized workflows,                                                                  CPU, built-in GPU, and NPU; processing what
  SmolVLA, GR00T, LeRobot and Pi0.5                                                    reference applications and demos, with                                                             each engine does best automatically
  with support for Hugging Face                                                        SLAM and control libraries
                                                                                                                                                                                        • Sustain deterministic control loops with
• Low-code visual interface: build physical                                          • Hardware integration: eliminate need to
                                                                                                                                                                                          asynchronous and real-time chunking even as
  AI workflows in an intuitive 3D                                                      build foundational plumbing from scratch
                                                                                                                                                                                          model complexity scales
  experience                                                                           with support for DevKits, Inputs



              Intel, the Intel logo, and other Intel marks are trademarks of I ntel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.                                          @2026 Intel Corporation   11
```

## Page 12

```text
  Physical AI Studio
  Open-source, end-to-end framework to simplify development of Intel-optimized Physical AI models.
  Leverages emerging techniques, e.g., imitation learning, and streamlines the entire VLA lifecycle —
  from robot calibration and data collection, through model fine-tuning, to optimized model export for
  inference on Intel.

           Key Features                                                                                                               Supported Policies                 Supported Robots
       • Data collection based on imitation learning                                                                                  • ACT                              • Trossen WidowX AI
       • Simple / modular API and CLI for training,                                                                                   • SmolVLA                          • SO-ARM101
         inference, and benchmarking Vision-                                                                                          • Pi0, Pi0.5                       • AlohaMini
         Language-Action policies
                                                                                                                                      • GR00T N1                         • LeKiwi
       • Export models to OpenVINO, ONNX, or
         Torch formats for accelerated inference                                                                                      • And many more                    • And many more
       • Benchmark policies on standardized
         simulation environments like LIBERO and
         PushT
       • Unified inference API across all export
         backends

Learn more: https://github.com/open-edge-platform/physi cal-ai-studio/
Intel, the Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.


                                                                                                                                                                                               @2026 Intel Corporation   12
```

## Page 13

```text
OpenVino                                Physical AI Framework
Build once, deploy everywhere. Brought to Physical AI inference and deployment.




     Physical AI Studio/LeRobot/Other workflows
          Robot Calibration                  Dataset Collection                   VLA Training           Optimization               Export




     OpenVINO Physical AI Framework
          Robot State                Camera               Observation             Preprocess          Infer          Postprocess       Robot Action



   Reducing Deployment Brittleness Across Industry
      •    Unifying cameras, robots, safety in one runtime                             •   Model predictive control loop, diffusion-based action
      •    Safety not as afterthought - Action clamping,                                   generation, multi-modal pipelines as modular plugins
           velocity limits, emergency stops, etc. built into the                           to runtime
           framework.                                                                  •   Enabling deployments to leverage hybrid, as well as
                                                                                           asynchronous compute components

                                                                                                                                        @2026 Intel Corporation   13
```

## Page 14

```text
Robotics AI Suite
Adapt and scale physical AI on Intel® for fast time to value



  Accelerate development for all kinds of robotics                               SDKs Content
  Industrial, service, mobile and humanoid robotics
  Locomotion, Perception, and Imitation Learning
                                                                                 Vision libraries
  Run vision AI, media analytics, and Gen AI workloads with real-time controls
                                                                                 Real-Time Control Frameworks

                                                                                 AI Inference Engines
  Innovate and scale fast with an open ecosystem
                                                                                 Orchestration-Ready Modules
  Validated systems, SDK, and vision AI use cases for accelerate time to value
                                                                                 Hardware-Aware Tuning
  Near zero-touch deployment and remote device management
  Rely on Intel’s proven long-standing Robotics Expertise                        ROS 2 & Open Standards




  Reduce Complexity, Improve TCO
  Latest generation processors for best possible performance
  Integrated CPUs, GPUs, NPUs for robotics performance across form factors
  Drive real-time controls & AI/perception from one platform



                                                                                                    @2026 Intel Corporation   14
```

## Page 15

```text
                                        Up to 180 TOPS



                                                                 Fast Response
                                                                   ▪ Ideal for low-latency AI workloads via OpenVINO
                                                                   ▪ Intel Time Coordinated Computing:
                                                                       – Resource partitioning (cache/mem)
                                               CPU                     – Deterministic power management
                                                                       – Cross-silicon time synch (Intel® TCC+TSN)


Intel® Core                                                      Low Power

Ultra Processors                                                   ▪ Ideal for sustained, complex AI workloads with
                                                                     excellent AI perf/watt efficiency
All-in-one Compute Platform for               NPU                  ▪ 50 TOPS AI compute
real-time control + AI workloads                                   ▪ Fast deployment via OpenVINO


                                                                 High Throughput
                                                                   ▪ Up to 12 Xe3 Cores
                                                                   ▪ AI deployment via OpenVINO
                                              GPU                  ▪ Computing acceleration via oneAPI

                                   1 Performance varies by use, configuration, and other factors. Learn more at intel.com/processor

                                   claims: Intel® Core Ultra processors, Edge. Results may vary.

                                                                                                                      @2026 Intel Corporation   15
```

