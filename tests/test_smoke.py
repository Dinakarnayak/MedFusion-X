import torch
from medfusion_x.ablation import ImageOnlyModel, TextOnlyModel
from medfusion_x.models import AdaptiveFusion, MedFusionX

class FakeImageEncoder(torch.nn.Module):
    def __init__(self,dim=32):
        super().__init__(); self.proj=torch.nn.Linear(16,dim)
    def forward(self,x):
        return torch.nn.functional.normalize(self.proj(x),dim=-1)

class FakeTextEncoder(torch.nn.Module):
    def __init__(self,dim=32):
        super().__init__(); self.proj=torch.nn.Linear(8,dim)
    def forward(self,texts):
        values=torch.tensor([[float((sum(map(ord,t))+i)%17)/17 for i in range(8)] for t in texts],dtype=torch.float32)
        return torch.nn.functional.normalize(self.proj(values),dim=-1)

def test_full_fusion_forward_backward():
    torch.manual_seed(42)
    model=MedFusionX(FakeImageEncoder(),FakeTextEncoder(),num_labels=14,fusion_dim=32,hidden=16)
    x=torch.randn(4,16); y=torch.randint(0,2,(4,14)).float()
    out=model(x,["normal chest xray"]*4)
    assert out["logits"].shape==(4,14); assert out["uncertainty"].shape==(4,1)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(out["logits"],y); loss.backward()
    assert any(p.grad is not None for p in model.parameters())

def test_ablation_models_forward():
    x=torch.randn(3,16)
    image=ImageOnlyModel(FakeImageEncoder(),dim=32)
    text=TextOnlyModel(FakeTextEncoder(),dim=32)
    assert image(x,["x"]*3)["logits"].shape==(3,14)
    assert text(x,["x"]*3)["logits"].shape==(3,14)

def test_adaptive_fusion_gates():
    fusion=AdaptiveFusion(dim=32,hidden_dim=32)
    fused,gi,gt=fusion(torch.randn(2,32),torch.randn(2,32))
    assert fused.shape==(2,32); assert gi.shape==gt.shape==(2,32)
    assert torch.all((gi>=0)&(gi<=1)); assert torch.all((gt>=0)&(gt<=1))
