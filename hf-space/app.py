import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import spaces
import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from medfusion_x.data import CHEST14_LABELS
from medfusion_x.models import BiomedCLIPEncoder, MedGemmaEncoder, MedFusionX

IMAGE_MODEL = os.getenv("MEDFUSION_IMAGE_MODEL", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
TEXT_MODEL = os.getenv("MEDFUSION_TEXT_MODEL", "google/medgemma-4b-it")
CHECKPOINT = os.getenv("MEDFUSION_CHECKPOINT", "")
CHECKPOINT_REPO = os.getenv("MEDFUSION_CHECKPOINT_REPO", "dinakarnayak/MedFusion-X-Checkpoint")
CHECKPOINT_FILENAME = os.getenv("MEDFUSION_CHECKPOINT_FILENAME", "best.pt")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
])

MODEL = None
MODEL_DEVICE = None
MODEL_CHECKPOINT = None
MODEL_STARTUP_ERROR = None


def resolve_checkpoint():
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not configured in the Hugging Face Space secrets. "
            "Add a read token with access to the private checkpoint repository "
            f"'{CHECKPOINT_REPO}' and the gated model '{TEXT_MODEL}'."
        )

    try:
        if CHECKPOINT:
            path = Path(CHECKPOINT)
            if path.exists():
                return str(path)
            return hf_hub_download(
                repo_id=CHECKPOINT,
                filename=CHECKPOINT_FILENAME,
                token=token,
            )

        return hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=CHECKPOINT_FILENAME,
            token=token,
        )
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face authentication/checkpoint access failed. "
            f"Verify HF_TOKEN has read access to '{CHECKPOINT_REPO}' and "
            f"'{TEXT_MODEL}' is approved for your account. "
            f"Original error: {exc}"
        ) from exc


def load_model(device):
    global MODEL_CHECKPOINT
    checkpoint = resolve_checkpoint()
    image_encoder = BiomedCLIPEncoder(IMAGE_MODEL, device, freeze=True, output_dim=512).to(device)
    text_encoder = MedGemmaEncoder(TEXT_MODEL, device, freeze=True, output_dim=512).to(device)
    model = MedFusionX(
        image_encoder,
        text_encoder,
        num_labels=len(CHEST14_LABELS),
        fusion_dim=512,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model_state_dict", state.get("model", state))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    MODEL_CHECKPOINT = checkpoint
    return model


def initialize_model():
    """Load the trained model at module scope for ZeroGPU compatibility."""
    global MODEL, MODEL_DEVICE, MODEL_CHECKPOINT, MODEL_STARTUP_ERROR
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        MODEL = load_model(device)
        MODEL_DEVICE = device
        MODEL_STARTUP_ERROR = None
    except Exception as exc:
        MODEL = None
        MODEL_DEVICE = None
        MODEL_CHECKPOINT = None
        MODEL_STARTUP_ERROR = str(exc)


def metric_card(label, value, description):
    return (
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-desc">{description}</div></div>'
    )


def score_band(score):
    if score >= 0.70:
        return "High"
    if score >= 0.30:
        return "Moderate"
    return "Low"


def build_report(run_id, timestamp, device, prompt, ranked, uncertainty, disagreement, image_gate, text_gate, latency_ms):
    return {
        "schema_version": "medfusion-x.research-report.v1",
        "run_id": run_id,
        "timestamp_utc": timestamp,
        "task": "NIH ChestX-ray14-style 14-label multilabel classification",
        "models": {
            "vision": IMAGE_MODEL,
            "language": TEXT_MODEL,
            "fusion": "adaptive",
        },
        "input": {
            "structured_text_prompt": prompt,
            "image_supplied": True,
        },
        "outputs": {
            "ranked_findings": [
                {"rank": i, "label": label, "score": round(float(score), 8), "band": score_band(score)}
                for i, (label, score) in enumerate(ranked, start=1)
            ],
            "uncertainty": uncertainty,
            "modality_disagreement": disagreement,
            "mean_image_gate": image_gate,
            "mean_text_gate": text_gate,
        },
        "runtime": {
            "device": device,
            "latency_ms": round(latency_ms, 2),
            "checkpoint": MODEL_CHECKPOINT,
        },
        "research_notice": (
            "Model outputs are for research/education only. They are not clinical "
            "diagnoses or calibrated patient-level disease probabilities."
        ),
    }


@spaces.GPU(duration=60)
def predict(image: Image.Image, text: str, history):
    global MODEL, MODEL_DEVICE

    if image is None:
        raise gr.Error("Upload a chest X-ray image first.")

    started = time.perf_counter()
    run_id = f"MF-{uuid.uuid4().hex[:10].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if MODEL is None or MODEL_DEVICE != device:
        try:
            MODEL = load_model(device)
            MODEL_DEVICE = device
        except Exception as exc:
            raise gr.Error(
                "MedFusion-X model initialization failed. "
                "Check the Space secret HF_TOKEN, private checkpoint access, "
                "and gated MedGemma access. "
                f"Technical detail: {exc}"
            ) from exc

    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    prompt = text.strip() or "Chest X-ray findings: No Finding"

    with torch.inference_mode():
        out = MODEL(x, [prompt])

    if device == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.perf_counter() - started) * 1000.0
    probs = torch.sigmoid(out["logits"])[0].detach().cpu()

    uncertainty = float(out["uncertainty"][0].item())
    disagreement = float(out["disagreement"][0].item())
    image_gate = float(out["image_gate"][0].mean().item())
    text_gate = float(out["text_gate"][0].mean().item())

    ranked = sorted(
        zip(CHEST14_LABELS, probs.tolist()),
        key=lambda z: z[1],
        reverse=True,
    )
    top_label, top_score = ranked[0]

    table = [
        [rank, label, f"{score:.2%}", score_band(score)]
        for rank, (label, score) in enumerate(ranked, start=1)
    ]

    report = build_report(
        run_id,
        timestamp,
        device,
        prompt,
        ranked,
        uncertainty,
        disagreement,
        image_gate,
        text_gate,
        latency_ms,
    )

    history = list(history or [])
    history.insert(
        0,
        [
            run_id,
            timestamp.replace("T", " ")[:19] + " UTC",
            top_label,
            f"{top_score:.2%}",
            f"{uncertainty:.4f}",
            f"{disagreement:.4f}",
            f"{latency_ms:.0f} ms",
        ],
    )
    history = history[:20]

    summary = f"""
### Inference result

**Top-ranked finding:** {top_label} · **{top_score:.2%}**

<div class="metric-grid">
{metric_card("Uncertainty", f"{uncertainty:.4f}", "Model uncertainty signal")}
{metric_card("Modality disagreement", f"{disagreement:.4f}", "Image/text probability disagreement")}
{metric_card("Image gate", f"{image_gate:.4f}", "Mean learned image contribution")}
{metric_card("Text gate", f"{text_gate:.4f}", "Mean learned text contribution")}
{metric_card("Latency", f"{latency_ms:.0f} ms", "End-to-end inference trace")}
{metric_card("Run ID", run_id, "Unique research run identifier")}
</div>

**Input prompt:** {prompt}

> These are model outputs for research analysis. They are not probabilities of disease in an individual patient and must not be used for diagnosis or treatment.
"""

    evidence = f"""
### Evidence & runtime diagnostics

| Signal | Value | Interpretation |
|---|---:|---|
| Top-ranked output | **{top_label}** | Highest model score |
| Top score | **{top_score:.2%}** | Model probability output |
| Uncertainty | **{uncertainty:.4f}** | Model uncertainty signal |
| Image/text disagreement | **{disagreement:.4f}** | Difference between modality predictions |
| Mean image gate | **{image_gate:.4f}** | Relative visual contribution |
| Mean text gate | **{text_gate:.4f}** | Relative textual contribution |
| Runtime | **{device.upper()}** | Active inference device |
| Latency | **{latency_ms:.0f} ms** | Measured request latency |
| Checkpoint | **Loaded** | Genuine trained checkpoint required |

**Research trace:** `{run_id}` · {timestamp}
"""

    warning = """
<div class="warning">
<b>Research / education only.</b><br>
MedFusion-X is not a clinical diagnostic system. Do not use these outputs
for medical decisions. Independently verify all findings with qualified
clinical expertise and validated clinical systems.
</div>
"""

    return summary, evidence, table, warning, history, report, f"● READY · {run_id} · {latency_ms:.0f} ms"


def clear_session():
    return [], [], None, "● READY · New research session"


css = """
body { background: #07111f; }
.gradio-container { max-width: 1320px !important; }
.hero { border: 1px solid #213852; border-radius: 18px; padding: 24px; background: linear-gradient(135deg,#0d1d31,#091321); }
.metric-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; }
.metric { border:1px solid #263f59; border-radius:12px; padding:14px; background:#0a1727; min-height:100px; }
.metric-label { color:#8fa7bd; font-size:10px; text-transform:uppercase; letter-spacing:1px; }
.metric-value { color:#5eead4; font-size:21px; font-weight:800; margin-top:4px; overflow-wrap:anywhere; }
.metric-desc { color:#8fa7bd; font-size:10px; margin-top:3px; }
.warning { border:1px solid #66531d; background:#211c0d; border-radius:12px; padding:14px; color:#e6d6a4; }
.livebar { border:1px solid #25445e; border-radius:10px; padding:10px 13px; background:#091725; color:#5eead4; font-family:monospace; }
@media(max-width:1100px){.metric-grid{grid-template-columns:repeat(3,1fr);}}
@media(max-width:700px){.metric-grid{grid-template-columns:1fr 1fr;}}
"""

with gr.Blocks(title="MedFusion-X | Live Research Inference", css=css) as demo:
    session_history = gr.State([])

    gr.HTML("""
    <div class="hero">
      <div style="color:#5eead4;font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase">
        MEDFUSION-X · LIVE RESEARCH INFERENCE · REAL-TIME TELEMETRY
      </div>
      <h1 style="margin:8px 0 4px">Evidence-aware multimodal analysis</h1>
      <p style="color:#9db0c4;margin:0">
        BiomedCLIP image evidence + MedGemma text representation + adaptive fusion,
        disagreement estimation, uncertainty diagnostics and reproducible inference traces.
      </p>
    </div>
    """)

    status = gr.Markdown("● READY · Waiting for a research run", elem_classes=["livebar"])

    with gr.Row():
        with gr.Column(scale=5):
            image = gr.Image(type="pil", label="Chest X-ray")
            text = gr.Textbox(
                label="Structured text prompt",
                value="Chest X-ray findings: No Finding",
                lines=3,
            )
            with gr.Row():
                run = gr.Button("▶ Run MedFusion-X", variant="primary")
                clear = gr.Button("↻ Clear session")
        with gr.Column(scale=7):
            summary = gr.Markdown(label="Inference summary")
            evidence = gr.Markdown(label="Evidence diagnostics")

    results = gr.Dataframe(
        headers=["Rank", "Pathology", "Model score", "Band"],
        datatype=["number", "str", "str", "str"],
        label="14-label ranked output",
        interactive=False,
    )

    warning = gr.HTML(
        '<div class="warning"><b>Research / education only.</b> Upload an image and run the model to view research outputs.</div>'
    )

    with gr.Accordion("Real-time run history · current browser session", open=True):
        history_table = gr.Dataframe(
            headers=[
                "Run ID",
                "UTC time",
                "Top finding",
                "Top score",
                "Uncertainty",
                "Disagreement",
                "Latency",
            ],
            datatype=["str"] * 7,
            interactive=False,
            value=[],
        )

    with gr.Row():
        with gr.Column():
            with gr.Accordion("Machine-readable research report", open=False):
                report = gr.JSON(label="Latest inference report")
        with gr.Column():
            with gr.Accordion("Model/runtime health", open=False):
                gr.Markdown(
                    f"**Vision encoder:** `{IMAGE_MODEL}`\n\n"
                    f"**Language encoder:** `{TEXT_MODEL}`\n\n"
                    f"**Checkpoint repository:** `{CHECKPOINT_REPO}`\n\n"
                    "**Checkpoint policy:** genuine trained weights required; no fabricated inference."
                )

    gr.Markdown(
        "### Interpretation note
"
        "A high model score is not a clinical diagnosis. Review the full ranked "
        "output together with uncertainty, modality disagreement, gates and the "
        "run trace. Session history is browser-session scoped and is not a patient record."
    )

    run.click(
        predict,
        inputs=[image, text, session_history],
        outputs=[summary, evidence, results, warning, history_table, report, status],
    ).then(
        lambda history: history,
        inputs=[history_table],
        outputs=[session_history],
    )

    clear.click(
        clear_session,
        outputs=[summary, evidence, history_table, status],
    ).then(
        lambda: [],
        outputs=[session_history],
    )

initialize_model()

if __name__ == "__main__":
    demo.launch()
