# Prometheus submission package

- `output/prometheus.pdf`: nine-slide pitch deck for the Junction form.
- `output/prometheus.pptx`: editable deck with speaker notes and evidence sources.
- `output/prometheus-pitch.mp4`: original narrated simulation pitch.
- `CRITERIA.md`: live form requirements, event rules and the conflicting official Intel rubrics.
- `form-fields.json`: draft submission copy. Link fields are updated during delivery.
- `PITCH-SCRIPT.md`: video narration, synthesized locally with the macOS Samantha voice.
- `assets/provenance.json`: source of each simulation visual and rollout metadata.
- `tools/`: rendering, deck-authoring and video assembly scripts.

Simulation demonstrations do not establish physical task success. The deck distinguishes implemented components, verified evidence and remaining validation.

The scripts were run in the development workspace. They require the existing MuJoCo/assets environment, the bundled Artifact Tool runtime, FFmpeg, macOS `say` and LibreOffice for PDF conversion. Build output is temporary. The presentation uses Arial and a 16:9 canvas.

## Updated physical demonstration

`output/prometheus-demo-3x.mp4` is the supplied physical footage at 3× speed (58.83 seconds). `output/prometheus-demo-narrated.mp4` is the updated 117.90-second pitch with normal-speed narration, the full physical run and a successful simulated pickup/placement clip. The current PDF/PPTX lead with this new physical evidence. The user identifies the physical controller as SmolVLM-assisted vision-guided execution, distinct from SmolVLA action-policy training. The footage reports three blue-plate sorts and a 48-second median cycle; this selected demo does not establish defect-branch reliability.

YouTube submission video (unlisted, viewable by anyone with the link): https://youtu.be/OFCFRsIQpCg
