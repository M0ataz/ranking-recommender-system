"""
Download and Parse MovieLens 100K Dataset
------------------------------------------
Downloads the MovieLens 100K dataset from GroupLens, parses the u.data file,
and saves it as data/interactions.csv with the required schema:
  user_id, item_id, interaction_count

The original u.data file has the format (tab-separated):
  user_id  item_id  rating  timestamp

We treat the rating value as the interaction_count, which serves as an
implicit feedback weight (higher rating = stronger preference signal).

Usage:
    python scripts/download_movielens.py [--output data/interactions.csv]

Dataset source:
    https://files.grouplens.org/datasets/movielens/ml-100k.zip
"""

import argparse
import io
import os
import sys
import zipfile

import pandas as pd
import requests

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "interactions.csv",
)


def download_movielens(output_path: str = DEFAULT_OUTPUT) -> None:
    """
    Download MovieLens 100K, parse u.data, and save to output_path as CSV.

    Args:
        output_path: Destination path for the parsed interactions CSV.
    """
    print(f"Downloading MovieLens 100K from:\n  {MOVIELENS_URL}\n")

    response = requests.get(MOVIELENS_URL, timeout=60)
    response.raise_for_status()
    print(f"Downloaded {len(response.content) / 1024:.1f} KB")

    # Extract u.data from the zip archive in memory
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        with zf.open("ml-100k/u.data") as f:
            raw = f.read().decode("utf-8")

    print("Parsing u.data ...")
    lines = [line.split("\t") for line in raw.strip().splitlines()]
    df = pd.DataFrame(lines, columns=["user_id", "item_id", "rating", "timestamp"])

    # Cast to integers
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(int)

    # Rename rating -> interaction_count (treat rating as implicit weight)
    df = df[["user_id", "item_id", "rating"]].rename(columns={"rating": "interaction_count"})

    # Aggregate duplicate (user, item) pairs by summing interaction counts
    df = (
        df.groupby(["user_id", "item_id"], as_index=False)["interaction_count"]
        .sum()
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"\nParsed dataset summary:")
    print(f"  Total interactions : {len(df):,}")
    print(f"  Unique users       : {df['user_id'].nunique():,}")
    print(f"  Unique items       : {df['item_id'].nunique():,}")
    print(f"  interaction_count  : min={df['interaction_count'].min()}, "
          f"max={df['interaction_count'].max()}, "
          f"mean={df['interaction_count'].mean():.2f}")
    print(f"\nSaved to: {output_path}")
    print("\nTo train with this data, run:")
    print(f"  python scripts/train.py --data-path {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and parse the MovieLens 100K dataset."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_movielens(output_path=args.output)
