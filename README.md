# MedFusion-X

PhD-level multimodal medical AI research pipeline combining BiomedCLIP image representations, MedGemma language representations, adaptive fusion, and uncertainty estimation.

Architecture:
Image -> BiomedCLIP -> image embedding
Text -> MedGemma -> text embedding
Image + Text -> adaptive gated fusion -> multi-label prediction
Unimodal predictions -> cross-modal disagreement -> uncertainty

Implemented:
- BiomedCLIP encoder
- MedGemma encoder
- Adaptive gated fusion
- Auxiliary unimodal heads
- Disagreement-aware uncertainty
- Multi-label training objective
- NIH ChestX-ray14 loader
- Reproducible splits and checkpoints
- AUROC, AUPRC, F1, precision, recall
- Selective-risk evaluation
- Unit tests
- Experiment protocol

Setup:
1. Create a Python 3.10+ environment.
2. Install with: pip install -e ".[dev]"
3. Authenticate with Hugging Face because MedGemma is gated.
4. Place ChestX-ray14 Data_Entry_2017.csv and images under data/nih_chest_xray14/.
5. Run: python scripts/train.py --config configs/default.yaml
6. Run: python scripts/evaluate.py --config configs/default.yaml

Important scientific qualification:
NIH ChestX-ray14 does not provide paired radiology reports. The default text modality is a label-derived structured prompt, which is useful for controlled ablation but is not genuine clinical-text multimodality. Use a paired image/report dataset for the clinical-text claim.

Results are intentionally not fabricated; all performance numbers must come from executed experiments.
