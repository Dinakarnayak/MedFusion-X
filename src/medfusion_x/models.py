import os
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class BiomedCLIPEncoder(nn.Module):
    def __init__(self, model_name, device, freeze=True, output_dim=512):
        super().__init__()
        import open_clip
        self.model, _, _ = open_clip.create_model_and_transforms("hf-hub:"+model_name)
        self.freeze=freeze
        if freeze:
            self.model.eval()
            for p in self.model.parameters(): p.requires_grad=False
        native_dim=getattr(self.model.visual,"output_dim",output_dim)
        self.proj=nn.Identity() if native_dim==output_dim else nn.Linear(native_dim,output_dim)

    def forward(self, images):
        ctx=torch.no_grad() if self.freeze else nullcontext()
        with ctx: z=self.model.encode_image(images, normalize=False)
        return F.normalize(self.proj(z.float()),dim=-1)

class MedGemmaEncoder(nn.Module):
    def __init__(self, model_name, device, freeze=True, output_dim=512, max_length=128):
        super().__init__()
        token=os.getenv("HF_TOKEN"); kwargs={"token":token} if token else {}
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,**kwargs)
        self.model=AutoModel.from_pretrained(model_name,**kwargs)
        self.device=device; self.freeze=freeze; self.max_length=max_length
        if freeze:
            self.model.eval()
            for p in self.model.parameters(): p.requires_grad=False
        hidden=self.model.config.hidden_size
        self.proj=nn.Identity() if hidden==output_dim else nn.Linear(hidden,output_dim)

    def forward(self,texts):
        batch=self.tokenizer(list(texts),padding=True,truncation=True,max_length=self.max_length,return_tensors="pt").to(self.device)
        ctx=torch.no_grad() if self.freeze else nullcontext()
        with ctx: out=self.model(**batch)
        h=out.last_hidden_state; mask=batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        pooled=(h*mask).sum(1)/mask.sum(1).clamp_min(1)
        return F.normalize(self.proj(pooled.float()),dim=-1)

class AdaptiveFusion(nn.Module):
    def __init__(self,dim=512,hidden_dim=512,dropout=0.2,temperature=1.0):
        super().__init__(); self.temperature=temperature
        self.image_gate=nn.Sequential(nn.Linear(dim*2,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,dim))
        self.text_gate=nn.Sequential(nn.Linear(dim*2,hidden_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden_dim,dim))
        self.fuse=nn.Sequential(nn.Linear(dim*2,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU(),nn.Dropout(dropout))
    def forward(self,image_z,text_z):
        pair=torch.cat([image_z,text_z],-1)
        gi=torch.sigmoid(self.image_gate(pair)/self.temperature); gt=torch.sigmoid(self.text_gate(pair)/self.temperature)
        fused=self.fuse(torch.cat([gi*image_z,gt*text_z],-1))
        return fused,gi,gt

class UncertaintyHead(nn.Module):
    def __init__(self,dim=512,hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim+2,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,fused,image_conf,text_conf):
        return F.softplus(self.net(torch.cat([fused,image_conf,text_conf],-1)))

class MedFusionX(nn.Module):
    def __init__(self,image_encoder,text_encoder,num_labels=14,fusion_dim=512,hidden=128,dropout=0.2):
        super().__init__(); self.image_encoder=image_encoder; self.text_encoder=text_encoder
        self.fusion=AdaptiveFusion(fusion_dim,fusion_dim,dropout)
        self.image_head=nn.Linear(fusion_dim,num_labels); self.text_head=nn.Linear(fusion_dim,num_labels)
        self.classifier=nn.Linear(fusion_dim,num_labels); self.uncertainty=UncertaintyHead(fusion_dim,hidden)

    def forward(self,images,texts):
        image_z=self.image_encoder(images); text_z=self.text_encoder(texts)
        fused,gi,gt=self.fusion(image_z,text_z)
        image_logits=self.image_head(image_z); text_logits=self.text_head(text_z); logits=self.classifier(fused)
        image_conf=torch.sigmoid(image_logits).mean(-1,keepdim=True); text_conf=torch.sigmoid(text_logits).mean(-1,keepdim=True)
        disagreement=torch.mean(torch.abs(torch.sigmoid(image_logits)-torch.sigmoid(text_logits)),dim=-1,keepdim=True)
        uncertainty=self.uncertainty(fused,image_conf,text_conf)+disagreement
        return {"logits":logits,"image_logits":image_logits,"text_logits":text_logits,
                "uncertainty":uncertainty,"image_gate":gi,"text_gate":gt,"disagreement":disagreement}
