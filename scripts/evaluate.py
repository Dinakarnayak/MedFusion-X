import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from medfusion_x.config import load_config
from medfusion_x.data import ChestXray14Dataset
from medfusion_x.models import BiomedCLIPEncoder, MedGemmaEncoder, MedFusionX
from medfusion_x.metrics import multilabel_metrics, selective_risk
from medfusion_x.utils import save_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/default.yaml"); ap.add_argument("--checkpoint",default="outputs/best.pt"); a=ap.parse_args()
    cfg=load_config(a.config); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx=torch.load(Path(cfg.evaluation["output_dir"])/"test_indices.pt",weights_only=True)
    d=cfg.data
    ds=ChestXray14Dataset(d["csv"],d["image_dir"],text_column=d["text_column"],image_column=d["image_column"],label_column=d["label_column"],indices=idx)
    loader=DataLoader(ds,batch_size=cfg.training["batch_size"],shuffle=False,num_workers=d["num_workers"])
    ie=BiomedCLIPEncoder(cfg.models["image_model"],device,True,cfg.models["embedding_dim"]).to(device)
    te=MedGemmaEncoder(cfg.models["text_model"],device,True,cfg.models["embedding_dim"],cfg.models["max_text_length"]).to(device)
    model=MedFusionX(ie,te,fusion_dim=cfg.models["embedding_dim"],hidden=cfg.fusion["uncertainty_hidden"],dropout=cfg.fusion["dropout"]).to(device)
    model.load_state_dict(torch.load(a.checkpoint,map_location=device)); model.eval()
    ys=[]; ps=[]; us=[]
    with torch.no_grad():
        for b in loader:
            o=model(b["image"].to(device),b["text"]); ys.append(b["target"]); ps.append(torch.sigmoid(o["logits"]).cpu()); us.append(o["uncertainty"].cpu())
    y=torch.cat(ys).numpy(); p=torch.cat(ps).numpy(); u=torch.cat(us).numpy().ravel()
    result=multilabel_metrics(y,p,cfg.evaluation["threshold"])
    for c in [0.5,0.7,0.8,0.9]: result[f"selective_risk_{int(c*100)}"]=selective_risk(y,p,u,c,cfg.evaluation["threshold"])
    save_json(result,Path(cfg.evaluation["output_dir"])/"test_metrics.json"); print(result)

if __name__=="__main__": main()
