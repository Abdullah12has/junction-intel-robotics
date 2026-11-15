# Prometheus submission package

- `output/prometheus.pdf`: nine-slide pitch deck for the Junction form.
- `output/prometheus.pptx`: editable deck with speaker notes and evidence sources.
- `output/prometheus-pitch.mp4`: narrated 98-second pitch, with simulation footage clearly identified.
- `CRITERIA.md`: live form requirements, event rules and the conflicting official Intel rubrics.
- `form-fields.json`: draft submission copy. Link fields are updated during delivery.
- `PITCH-SCRIPT.md`: video narration, synthesized locally with the macOS Samantha voice.
- `assets/provenance.json`: source of each simulation visual and rollout metadata.
- `tools/`: rendering, deck-authoring and video assembly scripts.

Simulation demonstrations do not establish physical task success. The deck distinguishes implemented components, verified evidence and remaining validation.

The scripts were run in the development workspace. They require the existing MuJoCo/assets environment, the bundled Artifact Tool runtime, FFmpeg, macOS `say` and LibreOffice for PDF conversion. Build output is temporary. The presentation uses Arial and a 16:9 canvas.
