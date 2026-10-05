# MedFusion-X 🩺🧠

**Uncertainty-aware multimodal medical AI for chest X-ray classification.**

MedFusion-X is a research-oriented framework that investigates whether **adaptive fusion of medical vision and language representations** can improve multilabel chest X-ray classification while exposing **modality disagreement, predictive uncertainty, and selective prediction performance**.

> **Research status:** Experimental / PhD-level research prototype. This project is not a clinical diagnostic system.

## 🚀 Live Research Demo

A ZeroGPU-compatible Gradio application is included at the repository root:

- **Live Space:** https://huggingface.co/spaces/dinakarnayak/MedFusion-X-Live
- **Space app:** `app.py`
- **Space dependencies:** `requirements.txt`

The live application accepts a chest X-ray and structured text prompt and, **only when a trained `best.pt` checkpoint is configured**, reports the 14 ChestX-ray14 pathology probabilities together with uncertainty, modality disagreement, and adaptive image/text gates.

The application intentionally produces **no fabricated predictions** when a trained checkpoint is unavailable.

## 🔬 Research Question

> **Does uncertainty-aware adaptive multimodal fusion outperform strong image-only and text-only baselines on ChestX-ray14, and can uncertainty estimates identify predictions that should be deferred or reviewed?**

The project is designed around a controlled comparison:

| Experiment | Image | Text | Adaptive Fusion | Purpose |
|---|:---:|:---:|:---:|---|
| **Image-only** | ✅ | ❌ | ❌ | Vision baseline |
| **Text-only** | ❌ | ✅ | ❌ | Language baseline |
| **Adaptive Fusion** | ✅ | ✅ | ✅ | Proposed multimodal model |

Each experiment writes to its own output directory so results can be compared without overwriting checkpoints.

## 🧠 Architecture

```text
                    Chest X-ray Image
                           │
                     ┌─────▼─────┐
                     │ BiomedCLIP│
                     └─────┬─────┘
                           │
                    Image embedding
                           │
                           ▼
                    Adaptive Fusion ◄──── MedGemma
                           │                  ▲
                    Fused representation     │
                           │             Text embedding
                           ▼
                 ┌────────────────────┐
                 │ Uncertainty Head   │
                 │ + Disagreement     │
                 └─────────┬──────────┘
                           │
                           ▼
                    14-label outputs
```

### Components

- **BiomedCLIP** — medical vision representation.
- **MedGemma** — medical language representation.
- **AdaptiveFusion** — learnable modality-specific gating.
- **Auxiliary heads** — image and text supervision.
- **Disagreement module** — measures divergence between modality predictions.
- **Uncertainty head** — predicts sample-level uncertainty.
- **Selective prediction** — evaluates performance at different coverage levels.

## 🩻 Dataset

The default experiment targets **NIH ChestX-ray14**, using its 14 pathology labels:

- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Effusion
- Emphysema
- Fibrosis
- Hernia
- Infiltration
- Mass
- Nodule
- Pleural Thickening
- Pneumonia
- Pneumothorax

### Important methodological limitation

ChestX-ray14 provides **image-level pathology labels**, not paired free-text radiology reports.

Therefore, unless a genuine report column is supplied, MedFusion-X constructs a deterministic structured prompt such as:

```text
Chest X-ray findings: Atelectasis, Effusion
```

This means the default experiment is a **controlled multimodal label-prompt study**, not a claim that ChestX-ray14 contains paired clinical reports.

## 📁 Repository Structure

```text
MedFusion-X/
├── app.py                    # ZeroGPU-compatible Gradio demo
├── requirements.txt          # Live Space dependencies
├── configs/
├── docs/
├── scripts/
├── src/
│   └── medfusion_x/
├── tests/
├── demo/
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── .github/
├── pyproject.toml
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/Dinakarnayak/MedFusion-X.git
cd MedFusion-X
pip install -e ".[dev]"
```

### Hugging Face authentication

MedGemma requires appropriate Hugging Face access.

Set the token locally without committing it:

```bash
export HF_TOKEN="your_token_here"
```

For Hugging Face Spaces, configure `HF_TOKEN` as a **Space Secret**, never as source code.

## 🗂️ Dataset Setup

The full NIH ChestX-ray14 image corpus is not committed to GitHub. Download it from the official NIH Clinical Center distribution and place the files locally under `data/nih_chest_xray14/`.

Expected structure:

```text
data/nih_chest_xray14/
├── Data_Entry_2017.csv
└── images/
    ├── 00000001_000.png
    └── ...
```

Validate the installation:

```bash
python scripts/prepare_dataset.py
```

## 🚀 Training

### Full adaptive-fusion experiment

```bash
python scripts/train.py --config configs/default.yaml
```

### Run individual experiments

```bash
python scripts/run_experiments.py --only image_only
python scripts/run_experiments.py --only text_only
python scripts/run_experiments.py --only adaptive_fusion
```

### Complete experiment matrix

```bash
python scripts/run_experiments.py
```

Outputs:

```text
outputs/
├── image_only/
├── text_only/
└── adaptive_fusion/
```

## 📊 Evaluation

```bash
python scripts/evaluate.py --config configs/default.yaml
```

The evaluation pipeline reports:

- Macro AUROC
- Macro AUPRC
- Micro F1
- Macro F1
- Precision
- Recall
- Per-class AUROC
- Per-class AUPRC
- Selective risk
- Uncertainty-aware coverage analysis

No performance numbers are hard-coded. Results must come from actual experiments.

## 🧪 Testing

```bash
pytest -q tests/test_smoke.py
python -m compileall -q src scripts tests
```

GitHub Actions runs compilation and unit tests automatically.

## 🔬 Reproducibility

The project includes:

- Fixed random seeds
- Deterministic train/validation/test index generation
- YAML experiment configurations
- Isolated experiment output directories
- Best-model checkpointing
- Training-history JSON
- Test-index persistence
- Automated CI testing
- Lightweight smoke tests
- Explicit dataset and methodological limitations

## 📈 Planned Research Evaluation

1. **Baseline comparison** — image-only vs text-only vs adaptive fusion.
2. **Ablation studies** — gating, auxiliary supervision, disagreement, uncertainty calibration, and encoder freezing.
3. **Uncertainty analysis** — calibration curves, reliability diagrams, ECE, selective risk, and high-disagreement cases.
4. **Statistical evaluation** — bootstrap confidence intervals, repeated seeds, and paired comparisons.
5. **Interpretability** — image attribution, modality gates, and disagreement case studies.
6. **Robustness** — distribution shift, missing modalities, image corruption, and text perturbation.

## ⚠️ Limitations and Responsible Use

MedFusion-X is a research prototype.

It should **not** be used to make clinical diagnoses, treatment decisions, or patient-management decisions.

Important limitations include:

- The default text modality is derived from labels rather than genuine radiology reports.
- Medical foundation models can produce incorrect or misleading representations.
- Dataset bias and label noise may affect results.
- Uncertainty estimates are not automatically equivalent to clinical confidence.
- Performance on one dataset does not establish generalisation to clinical populations.
- Independent validation is required before any clinical interpretation.

## 🛠️ Technology Stack

**Language:** Python

**Deep Learning:** PyTorch, Transformers, OpenCLIP

**Medical AI:** BiomedCLIP, MedGemma

**Data / Evaluation:** NumPy, pandas, scikit-learn, SciPy

**Engineering:** YAML, pytest, GitHub Actions, reproducible experiment outputs, Gradio, Hugging Face Spaces

## 📚 Research Documentation

- [Experiment design](docs/EXPERIMENTS.md)
- [Default configuration](configs/default.yaml)
- [Experiment configurations](configs/experiments/)
- [Source code](src/medfusion_x/)
- [Tests](tests/)
- [Live Gradio demo](https://huggingface.co/spaces/dinakarnayak/MedFusion-X-Live)

## 👨‍🔬 Research Direction

MedFusion-X is being developed as a research platform for investigating:

> **Multimodal medical AI + adaptive fusion + uncertainty estimation + selective prediction**

The long-term objective is to establish a rigorous experimental framework where multimodal performance is evaluated not only by classification accuracy, but also by **calibration, uncertainty quality, robustness, interpretability, and statistical significance**.

## 📄 Citation

```text
MedFusion-X
Dinakar Nayak N
Multimodal Medical AI Research Framework
https://github.com/Dinakarnayak/MedFusion-X
```

## ⚖️ License

See the repository license for usage terms.

---

**Built for reproducible medical AI research — not clinical deployment.**
