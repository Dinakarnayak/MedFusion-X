from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    raw: dict
    @property
    def data(self): return self.raw["data"]
    @property
    def models(self): return self.raw["models"]
    @property
    def fusion(self): return self.raw["fusion"]
    @property
    def training(self): return self.raw["training"]
    @property
    def evaluation(self): return self.raw["evaluation"]
    @property
    def experiments(self): return self.raw["experiments"]

def load_config(path: str) -> Config:
    with open(Path(path), "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
