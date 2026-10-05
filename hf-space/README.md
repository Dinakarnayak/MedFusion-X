---
title: MedFusion-X Live
emoji: 🧬
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
python_version: "3.12"
startup_duration_timeout: 30m
---

# MedFusion-X Live

Hugging Face ZeroGPU deployment for the MedFusion-X multimodal research inference app.

For the Space, upload `app.py`, this file, and `hf-space/requirements.txt` renamed to `requirements.txt` at the Space root.

The Space requirements install the GitHub package so the `src/medfusion_x` package is available to `app.py`.
