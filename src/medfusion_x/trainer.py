from pathlib import Path
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from .losses import total_loss
from .metrics import multilabel_metrics, selective_risk
from .utils import save_json, ensure_dir

class Trainer:
    def __init__(self,model,optimizer,device,cfg):
        self.model=model; self.optimizer=optimizer; self.device=device; self.cfg=cfg
        self.scaler=GradScaler("cuda",enabled=cfg["amp"] and device.type=="cuda"); self.best=-float("inf")

    def _epoch(self,loader,train=True):
        self.model.train(train); total=0; ys=[]; ps=[]; us=[]
        for b in tqdm(loader,leave=False):
            images=b["image"].to(self.device); targets=b["target"].to(self.device)
            if train: self.optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=self.device.type,enabled=self.scaler.is_enabled()):
                o=self.model(images,b["text"]); loss=total_loss(o,targets)
            if train:
                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg["grad_clip"])
                self.scaler.step(self.optimizer); self.scaler.update()
            total+=loss.item()*images.size(0); ys.append(targets.cpu()); ps.append(torch.sigmoid(o["logits"]).detach().cpu()); us.append(o["uncertainty"].detach().cpu())
        y=torch.cat(ys).numpy(); p=torch.cat(ps).numpy(); u=torch.cat(us).numpy().ravel()
        m=multilabel_metrics(y,p); m["loss"]=total/len(loader.dataset); m["selective_risk_80"]=selective_risk(y,p,u,0.8)["selective_risk"]
        return m

    def fit(self,train_loader,val_loader,epochs,patience,out_dir):
        out_dir=ensure_dir(out_dir); history=[]; stale=0
        for epoch in range(1,epochs+1):
            tr=self._epoch(train_loader,True); va=self._epoch(val_loader,False)
            history.append({"epoch":epoch,"train":tr,"val":va}); score=va["macro_auprc"]
            print(f"epoch={epoch} train_loss={tr['loss']:.4f} val_auprc={score:.4f}")
            if score>self.best:
                self.best=score; stale=0; torch.save(self.model.state_dict(),Path(out_dir)/"best.pt")
            else:
                stale+=1
                if stale>=patience: break
        save_json(history,Path(out_dir)/"history.json"); return history
