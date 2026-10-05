# MedFusion-X 🩺🧠

**Uncertainty-aware multimodal medical AI for chest X-ray classification.**

MedFusion-X is a research-oriented framework that investigates whether **adaptive fusion of medical vision and language representations** can improve multilabel chest X-ray classification while exposing **modality disagreement, predictive uncertainty, and selective prediction performance**.

> **Research status:** Experimental / PhD-level research prototype. This project is not a clinical diagnostic system.

---

## 🔬 Research Question

> **Does uncertainty-aware adaptive multimodal fusion outperform strong image-only and text-only baselines on ChestX-ray14, and can uncertainty estimates identify predictions that should be deferred or reviewed?**

The project is designed around a controlled comparison:

| Experiment | Image | Text | Adaptive Fusion | Purpose |
|---|:---:|:---:|:---:|---|
| **Image-only** | ✅ | ❌ | ❌ | Vision baseline |
| **Text-only** | ❌ | ✅ | ❌ | Language baseline |
| **Adaptive Fusion** | ✅ | ✅ | ✅ | Proposed multimodal model |

Each experiment writes to its own output directory so results can be compared without overwriting checkpoints.

---

## 🧠 Architecture

```text
                    ┌─────────────────────┐
                    │   Chest X-ray Image │
                    └──────────┬──────────┘
                               │
                       ┌───────▼────────┐
                       │   BiomedCLIP   │
                       └───────┬────────┘
                               │
                         Image embedding
                               │
                               │
                               ▼
                       ┌────────────────┐
                       │                │
                       │ AdaptiveFusion │──────► Image gate
                       │                │
                       └───────┬────────┘
                               │
                               │ Fused representation
                               │
     ┌─────────────────────────┘
     │
     │     ┌────────────────────────────┐
     └────►│       Uncertainty Head     │
           │ + modality disagreement    │
           └─────────────┬──────────────┘
                         │
                         ▼
                 Uncertainty estimate

   Structured text prompt
            │
            ▼
      ┌─────────────┐
      │  MedGemma   │
      └──────┬──────┘
             │
       Text embedding
             │
             └──────────────► AdaptiveFusion
                                   │
                                   ▼
                         Multilabel classifier
                                   │
                                   ▼
                         14 pathology outputs
```

### Components

- **BiomedCLIP** — medical vision representation.
- **MedGemma 1.5 4B IT** — medical language representation.
- **AdaptiveFusion** — learnable modality-specific gating.
- **Auxiliary heads** — image and text supervision.
- **Disagreement module** — measures divergence between modality predictions.
- **Uncertainty head** — predicts sample-level uncertainty.
- **Selective prediction** — evaluates performance at different coverage levels.

---

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

For a stronger research study, a future experiment can replace these prompts with an independently sourced dataset containing genuine image-report pairs.

---

## 📁 Repository Structure

```text
MedFusion-X/
├── configs/
│   ├── default.yaml
│   └── experiments/
├── docs/
│   └── EXPERIMENTS.md
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_experiments.py
│   └── ...
├── src/
│   └── medfusion_x/
│       ├── data.py
│       ├── models.py
│       ├── losses.py
│       ├── metrics.py
│       ├── trainer.py
│       ├── ablation.py
│       └── utils.py
├── tests/
│   ├── test_fusion.py
│   └── test_smoke.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
└── README.md
```

---

## ⚙️ Installation

Clone the repository and install the development dependencies:

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

**Never commit `HF_TOKEN` or any other credential to GitHub.**

---

## 🗂️ Dataset Layout

Place the dataset locally as:

```text
data/
└── nih_chest_xray14/
    ├── Data_Entry_2017.csv
    └── images/
        ├── 00000001_000.png
        ├── 00000001_001.png
        └── ...
```

The full image corpus is intentionally **not committed to this repository** because ChestX-ray14 is a large medical dataset hosted by NIH outside GitHub. Obtain it from the official NIH repository, place it under `data/nih_chest_xray14/`, and run `python scripts/prepare_dataset.py` to validate the installation. The repository includes `data/README.md` with the expected layout. citeturn0search0turn0search9

---

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

### Run the complete experiment matrix

```bash
python scripts/run_experiments.py
```

Outputs are isolated:

```text
outputs/
├── image_only/
├── text_only/
└── adaptive_fusion/
```

---

## 📊 Evaluation

After training:

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

No performance numbers are hard-coded into this README. Results should be generated from actual experiments.

---

## 🧪 Testing

MedFusion-X includes dependency-light smoke tests that do not download the large medical models:

```bash
pytest -q tests/test_smoke.py
```

Compile the complete project:

```bash
python -m compileall -q src scripts tests
```

The GitHub Actions workflow also runs compilation and unit tests automatically on repository pushes and pull requests.

---

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

---

## 📈 Planned Research Evaluation

The next research layer is intended to evaluate:

1. **Baseline comparison**
   - Image-only vs text-only vs adaptive fusion.

2. **Ablation studies**
   - No adaptive gating.
   - No auxiliary supervision.
   - No disagreement term.
   - No uncertainty calibration.
   - Frozen vs partially trainable encoders.

3. **Uncertainty analysis**
   - Calibration curves.
   - Reliability diagrams.
   - Expected Calibration Error.
   - Selective risk / coverage curves.
   - High-disagreement case analysis.

4. **Statistical evaluation**
   - Bootstrap confidence intervals.
   - Per-pathology comparisons.
   - Repeated-seed experiments.
   - Paired model comparisons.

5. **Interpretability**
   - Image attribution / Grad-CAM-style visualisation.
   - Modality gate analysis.
   - Disagreement case studies.

6. **Robustness**
   - Distribution-shift evaluation.
   - Missing-modality experiments.
   - Corrupted-image experiments.
   - Text perturbation experiments.

---

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

---

## 🛠️ Technology Stack

**Language**

- Python

**Deep Learning**

- PyTorch
- Transformers
- OpenCLIP

**Medical AI**

- BiomedCLIP
- MedGemma

**Data / Evaluation**

- NumPy
- pandas
- scikit-learn
- SciPy

**Engineering**

- YAML configuration
- pytest
- GitHub Actions
- Reproducible experiment outputs

---

## 📚 Research Documentation

- [Experiment design](docs/EXPERIMENTS.md)
- [Default configuration](configs/default.yaml)
- [Experiment configurations](configs/experiments/)
- [Source code](src/medfusion_x/)
- [Tests](tests/)

---

## 👨‍🔬 Research Direction

MedFusion-X is being developed as a research platform for investigating:

> **Multimodal medical AI + adaptive fusion + uncertainty estimation + selective prediction**

The long-term objective is to establish a rigorous experimental framework where multimodal performance is evaluated not only by classification accuracy, but also by **calibration, uncertainty quality, robustness, interpretability, and statistical significance**.

---

## 📄 Citation

If this project is used in academic work, please cite the repository and the specific experiment configuration/checkpoint used.

```text
MedFusion-X
Dinakar Nayak N
Multimodal Medical AI Research Framework
https://github.com/Dinakarnayak/MedFusion-X
```

---

## ⚖️ License

See the repository license for usage terms.

---

**Built for reproducible medical AI research — not clinical deployment.**
