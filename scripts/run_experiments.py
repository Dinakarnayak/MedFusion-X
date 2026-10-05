import argparse, copy, subprocess
from pathlib import Path
import yaml

EXPERIMENTS = {
    "image_only": {"modalities": ["image"], "fusion_enabled": False},
    "text_only": {"modalities": ["text"], "fusion_enabled": False},
    "adaptive_fusion": {"modalities": ["image", "text"], "fusion_enabled": True},
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/default.yaml")
    ap.add_argument("--only",choices=list(EXPERIMENTS),default=None)
    args=ap.parse_args()
    base=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    names=[args.only] if args.only else list(EXPERIMENTS)
    out=Path("configs/experiments"); out.mkdir(parents=True,exist_ok=True)
    for name in names:
        cfg=copy.deepcopy(base); cfg["experiments"].update({"run_name":name,**EXPERIMENTS[name]})
        path=out/(name+".yaml"); path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
        print("Prepared:",path)
        subprocess.run(["python","scripts/train.py","--config",str(path)],check=True)

if __name__=="__main__":
    main()
