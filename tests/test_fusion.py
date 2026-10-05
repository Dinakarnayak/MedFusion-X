import torch
from medfusion_x.models import AdaptiveFusion, UncertaintyHead

def test_adaptive_fusion_shapes():
    x=torch.randn(4,512); y=torch.randn(4,512); f,gi,gt=AdaptiveFusion()(x,y)
    assert f.shape==(4,512) and gi.shape==gt.shape==(4,512)

def test_uncertainty_positive():
    out=UncertaintyHead()(torch.randn(3,512),torch.rand(3,1),torch.rand(3,1))
    assert out.shape==(3,1) and torch.all(out>0)
