# MedFusion-X Experimental Protocol

## Research question
Does adaptive multimodal fusion improve multi-label chest X-ray prediction over unimodal baselines while providing useful uncertainty estimates?

## Baselines
1. Image-only BiomedCLIP
2. Text-only MedGemma
3. Simple feature concatenation
4. Adaptive gated fusion
5. Adaptive fusion plus disagreement-aware uncertainty

## Metrics
Macro AUROC, Macro AUPRC, micro/macro F1, precision, recall, per-class AUROC/AUPRC, selective risk, modality gate statistics, and image/text prediction disagreement.

## Dataset qualification
ChestX-ray14 provides image-level pathology labels rather than paired free-text radiology reports. The default text representation is therefore a structured prompt generated from the labels. This is a controlled multimodal ablation and must not be described as genuine image-report multimodality. A true clinical-text experiment requires a paired image/report dataset; the loader accepts one through data.text_column.

## Reproducibility
Record the YAML configuration, random seed, train/validation/test indices, checkpoint, history, metric JSON, and dependency versions for every run.
