#!/usr/bin/env python3
"""Download one public ChestX-ray14 image for MedFusion-X smoke testing.

This downloads a single 1024x1024 PNG from a public read-only mirror of
ChestX-ray14. It does not download or redistribute the full dataset.
"""

from pathlib import Path
from urllib.request import urlopen

IMAGE_NAME = "00000003_005.png"
IMAGE_URL = (
    "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/"
    f"images_1024x1024/{IMAGE_NAME}"
)
OUTPUT = Path("data/samples") / IMAGE_NAME


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(IMAGE_URL, timeout=60) as response:
        data = response.read()
    OUTPUT.write_bytes(data)
    print(f"Downloaded {IMAGE_NAME} -> {OUTPUT} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
