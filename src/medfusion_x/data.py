from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CHEST14_LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
"Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
"Pleural_Thickening","Hernia"]

class ChestXray14Dataset(Dataset):
    """ChestX-ray14 loader. Default text is a structured label prompt, not a radiology report."""
    def __init__(self, csv_path, image_dir, image_transform=None, text_column=None,
                 image_column="Image Index", label_column="Finding Labels", indices=None):
        self.df = pd.read_csv(csv_path)
        if indices is not None: self.df = self.df.iloc[indices].reset_index(drop=True)
        self.image_dir = Path(image_dir); self.image_column=image_column
        self.label_column=label_column; self.text_column=text_column
        self.transform=image_transform or transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize((0.481,0.457,0.408),(0.268,0.261,0.275))])

    def __len__(self): return len(self.df)

    def _labels(self, raw):
        labels=[] if str(raw)=="No Finding" else [x.strip() for x in str(raw).split("|")]
        target=torch.zeros(len(CHEST14_LABELS), dtype=torch.float32)
        for label in labels:
            if label in CHEST14_LABELS: target[CHEST14_LABELS.index(label)]=1.0
        return target, labels

    def __getitem__(self, idx):
        row=self.df.iloc[idx]
        image=Image.open(self.image_dir / str(row[self.image_column])).convert("RGB")
        target, labels=self._labels(row[self.label_column])
        if self.transform: image=self.transform(image)
        if self.text_column and self.text_column in row and pd.notna(row[self.text_column]):
            text=str(row[self.text_column])
        else:
            text="Chest X-ray findings: "+(", ".join(labels) if labels else "No Finding")
        return {"image":image,"text":text,"target":target,"image_id":str(row[self.image_column])}
