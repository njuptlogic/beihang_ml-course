"""Fetch the instructor's archives with the downloader used for experiment 1."""
import argparse
from pathlib import Path

from experiment1 import download_data

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=int, choices=[2, 3])
    number = parser.parse_args().experiment
    prefix = {2: "2-MedImage", 3: "3-Saliency"}[number]
    download_data.ROOT = ROOT / f"experiment{number}"
    download_data.NAMES = [f"{prefix}-{part}Set.zip" for part in ("Train", "Test")]
    download_data.main()
