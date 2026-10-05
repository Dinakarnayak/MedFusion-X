import torch
import torch.nn.functional as F

def multilabel_loss(o,y):
    main=F.binary_cross_entropy_with_logits(o["logits"],y)
    image=F.binary_cross_entropy_with_logits(o["image_logits"],y)
    text=F.binary_cross_entropy_with_logits(o["text_logits"],y)
    consistency=F.mse_loss(torch.sigmoid(o["image_logits"]),torch.sigmoid(o["text_logits"]))
    return main+0.25*(image+text)+0.10*consistency

def total_loss(o,y):
    base=multilabel_loss(o,y)
    errors=(torch.sigmoid(o["logits"]).detach()-y).abs().mean(-1)
    calibration=F.mse_loss(o["uncertainty"].squeeze(-1),errors)
    return base+0.05*calibration
