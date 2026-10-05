---
title: MedFusion-X Live Inference
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# MedFusion-X Live Inference

Online inference frontend for MedFusion-X.

Configure `HF_TOKEN` as a Space secret when gated MedGemma access is required.

Set `MEDFUSION_CHECKPOINT` to a trained MedFusion-X `best.pt` checkpoint. The app intentionally refuses to produce pathology predictions without a trained checkpoint, preventing untrained classifier heads from being presented as medical predictions.

Environment variables:
- `MEDFUSION_IMAGE_MODEL`
- `MEDFUSION_TEXT_MODEL`
- `MEDFUSION_CHECKPOINT`

Research prototype only; not for clinical diagnosis or treatment decisions.
