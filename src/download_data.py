"""Helper that guarantees data/train.csv exists.

Usage (from the project root):

    python src/download_data.py

It downloads the standard 891 x 12 Kaggle training file from a public
mirror, then verifies the shape and the three known missing-value counts.
If your machine is offline, download train.csv manually from
https://www.kaggle.com/c/titanic/data and save it as data/train.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .pipeline import DATA_PATH, MIRROR_URL, load_titanic_data, missing_value_counts
except ImportError:
    from pipeline import DATA_PATH, MIRROR_URL, load_titanic_data, missing_value_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Titanic train.csv.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even if data/train.csv already exists.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA_PATH,
        help=f"Destination path (default: {DATA_PATH}).",
    )
    args = parser.parse_args()

    destination = args.out
    if destination.exists() and not args.force:
        print(f"Dataset already exists: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading: {MIRROR_URL}")
        import urllib.request

        urllib.request.urlretrieve(MIRROR_URL, destination)
        print(f"Saved to   : {destination}")

    df = load_titanic_data(destination, allow_download=False)
    print(f"Shape      : {df.shape[0]} rows x {df.shape[1]} columns")
    print("Missing    :")
    print(missing_value_counts(df).to_string())
    print("\nDataset is ready. Next step: python src/train_model.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
