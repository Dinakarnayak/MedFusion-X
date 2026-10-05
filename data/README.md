# NIH ChestX-ray14

The full NIH ChestX-ray14 image dataset is not stored in this Git repository.

ChestX-ray14 is a large medical imaging dataset released by the NIH Clinical Center and hosted outside GitHub. Obtain it directly from the dataset provider and follow the applicable data-use and attribution terms.

Official dataset repository:

https://nihcc.app.box.com/v/ChestXray-NIHCC

## Expected layout

data/
└── nih_chest_xray14/
    ├── Data_Entry_2017.csv
    └── images/
        ├── 00000001_000.png
        ├── 00000001_001.png
        └── ...

## Validate the installation

From the repository root:

    python scripts/prepare_dataset.py

The validator checks the metadata schema and image directory.

## Why the images are not committed to GitHub

The dataset contains a very large medical-image corpus and is distributed by NIH outside GitHub. This repository stores the research code, configuration, documentation, and validation tooling while the dataset is obtained directly from the provider.

Do not commit the downloaded image corpus to this repository.