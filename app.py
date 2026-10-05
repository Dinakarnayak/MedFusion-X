import os
from pathlib import Path

import spaces
import torch
import gradio as gr
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

def resolve_checkpoint():
    if CHECKPOINT:
        path = Path(CHECKPOINT)
        if path.exists():
            return str(path)
        return hf_hub_download(repo_id=CHECKPOINT, filename=CHECKPOINT_FILENAME, token=os.getenv("HF_TOKEN"))
    return hf_hub_download(repo_id=CHECKPOINT_REPO, filename=CHECKPOINT_FILENAME, token=os.getenv("HF_TOKEN"))

def load_model(device):
    checkpoint = resolve_checkpoint()
    image_encoder = BiomedCLIPEncoder(IMAGE_MODEL, device, freeze=True, output_dim=512).to(device)
    text_encoder = MedGemmaEncoder(TEXT_MODEL, device, freeze=True, output_dim=512).to(device)
    model = MedFusionX(image_encoder, text_encoder, num_labels=len(CHEST14_LABELS), fusion_dim=512).to(device)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model_state_dict", state.get("model", state))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

@spaces.GPU(duration=180)
def predict(image: Image.Image, text: str):
    global MODEL, MODEL_DEVICE
    if image is None:
        raise gr.Error("Upload a chest X-ray image first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if MODEL is None or MODEL_DEVICE != device:
        try:
            MODEL = load_model(device)
            MODEL_DEVICE = device
        except Exception as exc:
            raise gr.Error(
                "MedFusion-X could not load a trained checkpoint. "
                "Configure HF_TOKEN and upload best.pt to "
                f"{CHECKPOINT_REPO}, or set MEDFUSION_CHECKPOINT. "
                f"Technical detail: {exc}"
            ) from exc

    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    prompt = text.strip() or "Chest X-ray findings: No Finding"

    with torch.inference_mode():
        out = MODEL(x, [prompt])

    probs = torch.sigmoid(out["logits"])[0].detach().cpu()
    uncertainty = float(out["uncertainty"][0].item())
    disagreement = float(out["disagreement"][0].item())
    image_gate = float(out["image_gate"][0].mean().item())
    text_gate = float(out["text_gate"][0].mean().item())

    ranked = sorted(zip(CHEST14_LABELS, probs.tolist()), key=lambda z: z[1], reverse=True)
    top_label, top_score = ranked[0]

    table = [
        [rank, label, f"{score:.2%}", "High" if score >= 0.70 else "Moderate" if score >= 0.30 else "Low"]
        for rank, (label, score) in enumerate(ranked, start=1)
    ]

    summary = f"""
### Inference result

**Top-ranked finding:** {top_label} · **{top_score:.2%}**

<div class="metric-grid">
{metric_card("Uncertainty", f"{uncertainty:.4f}", "Model uncertainty signal")}
{metric_card("Modality disagreement", f"{disagreement:.4f}", "Image/text probability disagreement")}
{metric_card("Image gate", f"{image_gate:.4f}", "Mean learned image contribution")}
{metric_card("Text gate", f"{text_gate:.4f}", "Mean learned text contribution")}
</div>

**Input prompt:** {prompt}

> These are model outputs for research analysis. They are not probabilities of disease in an individual patient and must not be used for diagnosis or treatment.
"""

    evidence = f"""
### Evidence diagnostics

| Signal | Value | Interpretation |
|---|---:|---|
| Top-ranked output | **{top_label}** | Highest model score |
| Top score | **{top_score:.2%}** | Model probability output |
| Uncertainty | **{uncertainty:.4f}** | Model uncertainty signal |
| Image/text disagreement | **{disagreement:.4f}** | Difference between modality predictions |
| Mean image gate | **{image_gate:.4f}** | Relative visual contribution |
| Mean text gate | **{text_gate:.4f}** | Relative textual contribution |

**Model:** BiomedCLIP + MedGemma + adaptive fusion  
**Task:** NIH ChestX-ray14-style 14-label multilabel classification  
**Runtime:** {device.upper()}
"""

    warning = """
<div class="warning">
<b>Research / education only.</b><br>
MedFusion-X is not a clinical diagnostic system. Do not use these outputs
for medical decisions. Independently verify all findings with qualified
clinical expertise and validated clinical systems.
</div>
"""
    return summary, evidence, table, warning

def metric_card(label, value, description):
    return (
        f'<div class="metric"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-desc">{description}</div></div>'
    )

css = """
body { background: #07111f; }
.gradio-container { max-width: 1250px !important; }
.hero { border: 1px solid #213852; border-radius: 18px; padding: 24px; background: linear-gradient(135deg,#0d1d31,#091321); }
.metric-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; }
.metric { border:1px solid #263f59; border-radius:12px; padding:14px; background:#0a1727; }
.metric-label { color:#8fa7bd; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.metric-value { color:#5eead4; font-size:24px; font-weight:800; margin-top:4px; }
.metric-desc { color:#8fa7bd; font-size:11px; margin-top:3px; }
.warning { border:1px solid #66531d; background:#211c0d; border-radius:12px; padding:14px; color:#e6d6a4; }
@media(max-width:800px){.metric-grid{grid-template-columns:1fr 1fr;}}
"""

with gr.Blocks(title="MedFusion-X | Live Research Inference", css=css) as demo:
    gr.HTML("""
    <div class="hero">
      <div style="color:#5eead4;font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase">
        MEDFUSION-X · LIVE RESEARCH INFERENCE
      </div>
      <h1 style="margin:8px 0 4px">Evidence-aware multimodal analysis</h1>
      <p style="color:#9db0c4;margin:0">
        BiomedCLIP image evidence + MedGemma text representation + adaptive fusion,
        disagreement estimation and uncertainty diagnostics.
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=5):
            image = gr.Image(type="pil", label="Chest X-ray")
            text = gr.Textbox(label="Structured text prompt", value="Chest X-ray findings: No Finding", lines=3)
            run = gr.Button("▶ Run MedFusion-X", variant="primary")
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
    gr.Markdown(
        "### Interpretation note\n"
        "A high model score is not a clinical diagnosis. Review the full ranked "
        "output together with uncertainty and modality disagreement."
    )
    run.click(predict, inputs=[image, text], outputs=[summary, evidence, results, warning])

if __name__ == "__main__":
    demo.launch()
