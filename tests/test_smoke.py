import math
from pathlib import Path

import torch
from medfusion_x.ablation import ImageOnlyModel, TextOnlyModel
from medfusion_x.models import AdaptiveFusion, MedFusionX


class FakeImageEncoder(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.proj = torch.nn.Linear(16, dim)

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), dim=-1)


class FakeTextEncoder(torch.nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.proj = torch.nn.Linear(8, dim)

    def forward(self, texts):
        values = torch.tensor(
            [[float((sum(map(ord, t)) + i) % 17) / 17 for i in range(8)] for t in texts],
            dtype=torch.float32,
        )
        return torch.nn.functional.normalize(self.proj(values), dim=-1)


def assert_finite_tree(output):
    for key, value in output.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), f"{key} contains NaN/Inf"


def test_full_fusion_forward_backward():
    torch.manual_seed(42)
    model = MedFusionX(
        FakeImageEncoder(),
        FakeTextEncoder(),
        num_labels=14,
        fusion_dim=32,
        hidden=16,
    )
    x = torch.randn(4, 16)
    y = torch.randint(0, 2, (4, 14)).float()

    out = model(x, ["normal chest xray"] * 4)

    assert out["logits"].shape == (4, 14)
    assert out["image_logits"].shape == (4, 14)
    assert out["text_logits"].shape == (4, 14)
    assert out["uncertainty"].shape == (4, 1)
    assert out["disagreement"].shape == (4, 1)
    assert_finite_tree(out)
    assert torch.all(out["uncertainty"] >= 0)
    assert torch.all((out["disagreement"] >= 0) & (out["disagreement"] <= 1))

    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], y)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_ablation_models_forward():
    torch.manual_seed(7)
    x = torch.randn(3, 16)
    image = ImageOnlyModel(FakeImageEncoder(), dim=32)
    text = TextOnlyModel(FakeTextEncoder(), dim=32)

    image_out = image(x, ["x"] * 3)
    text_out = text(x, ["x"] * 3)

    assert image_out["logits"].shape == (3, 14)
    assert text_out["logits"].shape == (3, 14)
    assert_finite_tree(image_out)
    assert_finite_tree(text_out)


def test_adaptive_fusion_gates():
    torch.manual_seed(11)
    fusion = AdaptiveFusion(dim=32, hidden_dim=32)
    fused, gi, gt = fusion(torch.randn(2, 32), torch.randn(2, 32))

    assert fused.shape == (2, 32)
    assert gi.shape == gt.shape == (2, 32)
    assert torch.all((gi >= 0) & (gi <= 1))
    assert torch.all((gt >= 0) & (gt <= 1))
    assert torch.isfinite(fused).all()


def test_research_portal_contains_core_sections():
    html = Path("docs/index.html").read_text(encoding="utf-8")

    required_markers = [
        "Evidence-aware",
        "Research framing",
        "Architecture",
        "Experimental matrix",
        "Adaptive fusion",
        "Research console",
        "Reproducibility",
        "Model card",
        "Research integrity boundary",
        "MedFusion-X-Live",
    ]
    for marker in required_markers:
        assert marker in html, f"Missing portal marker: {marker}"

    assert 'data-tab="baseline"' in html
    assert 'data-tab="fusion"' in html
    assert 'data-tab="reliability"' in html
    assert 'data-tab="robustness"' in html


def test_research_portal_has_no_fabricated_metrics():
    html = Path("docs/index.html").read_text(encoding="utf-8").lower()

    # The portal should describe metrics as evaluation targets, not publish
    # unsupported benchmark values.
    forbidden_patterns = [
        "accuracy: 99",
        "auroc: 0.99",
        "auc: 0.99",
        "sensitivity: 99",
        "specificity: 99",
    ]
    for pattern in forbidden_patterns:
        assert pattern not in html


def test_uncertainty_is_numerically_valid():
    torch.manual_seed(123)
    model = MedFusionX(
        FakeImageEncoder(),
        FakeTextEncoder(),
        num_labels=14,
        fusion_dim=32,
        hidden=16,
    )
    with torch.no_grad():
        out = model(torch.randn(5, 16), ["controlled finding"] * 5)

    assert out["uncertainty"].min().item() >= 0
    assert all(math.isfinite(float(v)) for v in out["uncertainty"].flatten())
