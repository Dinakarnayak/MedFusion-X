import torch
import torch.nn as nn

class ImageOnlyModel(nn.Module):
    def __init__(self, encoder, dim=512, labels=14):
        super().__init__(); self.encoder=encoder; self.head=nn.Linear(dim,labels)
    def forward(self,images,texts=None):
        z=self.encoder(images); return {"logits":self.head(z),"image_logits":self.head(z),"text_logits":self.head(z),
          "uncertainty":torch.ones(images.size(0),1,device=images.device),"disagreement":torch.zeros(images.size(0),1,device=images.device)}

class TextOnlyModel(nn.Module):
    def __init__(self, encoder, dim=512, labels=14):
        super().__init__(); self.encoder=encoder; self.head=nn.Linear(dim,labels)
    def forward(self,images,texts):
        z=self.encoder(texts); logits=self.head(z)
        return {"logits":logits,"image_logits":logits,"text_logits":logits,
          "uncertainty":torch.ones(len(texts),1,device=logits.device),"disagreement":torch.zeros(len(texts),1,device=logits.device)}
