import argparse
from pathlib import Path
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from medfusion_x.ablation import ImageOnlyModel, TextOnlyModel
from medfusion_x.config import load_config
from medfusion_x.data import CHEST14_LABELS, ChestXray14Dataset
from medfusion_x.models import BiomedCLIPEncoder, MedGemmaEncoder, MedFusionX
from medfusion_x.trainer import Trainer
from medfusion_x.utils import seed_everything

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml"); a=ap.parse_args()
    cfg=load_config(a.config); seed_everything(cfg.raw["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d=cfg.data; df=pd.read_csv(d["csv"])
    if d["max_samples"]: df=df.iloc[:d["max_samples"]]
    idx=list(range(len(df)))
    train_idx,temp=train_test_split(idx,test_size=d["val_fraction"]+d["test_fraction"],random_state=cfg.raw["seed"])
    rel=d["test_fraction"]/(d["val_fraction"]+d["test_fraction"])
    val_idx,test_idx=train_test_split(temp,test_size=rel,random_state=cfg.raw["seed"])
    common=dict(csv_path=d["csv"],image_dir=d["image_dir"],text_column=d["text_column"],image_column=d["image_column"],label_column=d["label_column"])
    datasets=[ChestXray14Dataset(**common,indices=i) for i in [train_idx,val_idx,test_idx]]
    loaders=[DataLoader(ds,batch_size=cfg.training["batch_size"],shuffle=(i==0),num_workers=d["num_workers"],pin_memory=(device.type=="cuda")) for i,ds in enumerate(datasets)]
    modalities=set(cfg.raw.get("experiments",{}).get("modalities",["image","text"]))
    if modalities=={"image"}:
        image_enc=BiomedCLIPEncoder(cfg.models["image_model"],device,cfg.models["freeze_image_encoder"],cfg.models["embedding_dim"]).to(device)
        model=ImageOnlyModel(image_enc,cfg.models["embedding_dim"],len(CHEST14_LABELS)).to(device)
    elif modalities=={"text"}:
        text_enc=MedGemmaEncoder(cfg.models["text_model"],device,cfg.models["freeze_text_encoder"],cfg.models["embedding_dim"],cfg.models["max_text_length"]).to(device)
        model=TextOnlyModel(text_enc,cfg.models["embedding_dim"],len(CHEST14_LABELS)).to(device)
    else:
        image_enc=BiomedCLIPEncoder(cfg.models["image_model"],device,cfg.models["freeze_image_encoder"],cfg.models["embedding_dim"]).to(device)
        text_enc=MedGemmaEncoder(cfg.models["text_model"],device,cfg.models["freeze_text_encoder"],cfg.models["embedding_dim"],cfg.models["max_text_length"]).to(device)
        model=MedFusionX(image_enc,text_enc,fusion_dim=cfg.models["embedding_dim"],hidden=cfg.fusion["uncertainty_hidden"],dropout=cfg.fusion["dropout"]).to(device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=cfg.training["lr"],weight_decay=cfg.training["weight_decay"])
    out_dir=Path(cfg.evaluation["output_dir"]); out_dir.mkdir(parents=True,exist_ok=True)
    Trainer(model,optimizer,device,cfg.training).fit(loaders[0],loaders[1],cfg.training["epochs"],cfg.training["patience"],out_dir)
    torch.save(model.state_dict(),out_dir/"last.pt"); torch.save(test_idx,out_dir/"test_indices.pt")

if __name__=="__main__": main()
