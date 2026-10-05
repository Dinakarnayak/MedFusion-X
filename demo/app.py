import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from PIL import Image
from torchvision import transforms

from medfusion_x.data import CHEST14_LABELS
from medfusion_x.models import BiomedCLIPEncoder, MedGemmaEncoder, MedFusionX

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_MODEL = os.getenv("MEDFUSION_IMAGE_MODEL", "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
TEXT_MODEL = os.getenv("MEDFUSION_TEXT_MODEL", "google/medgemma-4b-it")
CHECKPOINT = os.getenv("MEDFUSION_CHECKPOINT", "")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),
])
MODEL = None

def load_model():
    if not CHECKPOINT or not Path(CHECKPOINT).exists():
        raise RuntimeError(
            "No trained MedFusion-X checkpoint is configured. "
            "Set MEDFUSION_CHECKPOINT to a trained best.pt checkpoint. "
            "No fabricated predictions are produced."
        )
    image_encoder = BiomedCLIPEncoder(IMAGE_MODEL, DEVICE, freeze=True, output_dim=512).to(DEVICE)
    text_encoder = MedGemmaEncoder(TEXT_MODEL, DEVICE, freeze=True, output_dim=512).to(DEVICE)
    model = MedFusionX(
        image_encoder, text_encoder,
        num_labels=len(CHEST14_LABELS), fusion_dim=512
    ).to(DEVICE)
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state.get("model_state_dict", state.get("model", state)), strict=True)
    model.eval()
    return model

@spaces.GPU(duration=180)
def predict(image: Image.Image, text: str):
    global MODEL
    if image is None:
        raise gr.Error("Upload a chest X-ray image first.")
    if MODEL is None:
        MODEL = load_model()

    x = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    prompt = text.strip() or "Chest X-ray findings: No Finding"

    with torch.inference_mode():
        out = MODEL(x, [prompt])

    probs = torch.sigmoid(out["logits"])[0].cpu()
    uncertainty = float(out["uncertainty"][0].item())
    disagreement = float(out["disagreement"][0].item())
    image_gate = float(out["image_gate"][0].mean().item())
    text_gate = float(out["text_gate"][0].mean().item())

    ranked = sorted(zip(CHEST14_LABELS, probs.tolist()), key=lambda z: z[1], reverse=True)
    table = [[label, round(score, 4)] for label, score in ranked]
    summary = (
        f"Top prediction: {ranked[0][0]} ({ranked[0][1]:.1%})\n"
        f"Uncertainty: {uncertainty:.4f}\n"
        f"Image/text disagreement: {disagreement:.4f}\n"
        f"Mean image gate: {image_gate:.4f}\n"
        f"Mean text gate: {text_gate:.4f}"
    )
    return summary, table, (
        "Research-only output. Not a clinical diagnostic system; "
        "independently verify all results."
    )

with gr.Blocks(title="MedFusion-X Live Inference") as demo:
    gr.Markdown(
        "# MedFusion-X — Live Inference\n"
        "Upload a chest X-ray and run the trained multimodal model."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Chest X-ray")
            text = gr.Textbox(
                label="Structured text prompt",
                value="Chest X-ray findings: No Finding",
                lines=2,
            )
            run = gr.Button("Run MedFusion-X", variant="primary")
        with gr.Column():
            summary = gr.Textbox(label="Inference summary", lines=7)
            results = gr.Dataframe(
                headers=["Pathology", "Probability"],
                datatype=["str", "number"],
                label="14 pathology outputs",
            )
    warning = gr.Markdown(
        "⚠️ Research/education only. Not for diagnosis or treatment decisions."
    )
    run.click(predict, [image, text], [summary, results, warning])

if __name__ == "__main__":
    demo.launch()
