"""
Training Script
---------------
Generates or loads implicit feedback data, trains the ALS model,
evaluates it with ranking metrics, and saves the model to disk.

Usage (synthetic data — default):
    python scripts/train.py

Usage (real data from CSV):
    python scripts/train.py --data-path data/interactions.csv

The CSV must have columns: user_id, item_id, interaction_count
(as produced by scripts/download_movielens.py).

Full options:
    python scripts/train.py \\
        --data-path data/interactions.csv \\
        --factors 128 \\
        --iterations 50 \\
        --k 5 10 20
"""

import argparse
import os
import sys
import json

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd

from src.data import generate_implicit_data, train_test_split_implicit
from src.model import ImplicitALSModel
from src.evaluate import evaluate_model, print_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the ranking recommender system.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data source ---
    data_group = parser.add_argument_group("Data source")
    data_group.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to a CSV file with columns [user_id, item_id, interaction_count]. "
            "If omitted, synthetic data is generated instead."
        ),
    )

    # --- Synthetic data options (only used when --data-path is not set) ---
    synth_group = parser.add_argument_group("Synthetic data (ignored when --data-path is set)")
    synth_group.add_argument("--num-users", type=int, default=1000)
    synth_group.add_argument("--num-items", type=int, default=5000)
    synth_group.add_argument("--num-interactions", type=int, default=50000)

    # --- Model hyperparameters ---
    model_group = parser.add_argument_group("Model hyperparameters")
    model_group.add_argument("--factors", type=int, default=128)
    model_group.add_argument("--regularization", type=float, default=0.01)
    model_group.add_argument("--iterations", type=int, default=50)
    model_group.add_argument("--alpha", type=float, default=40.0)

    # --- Evaluation ---
    eval_group = parser.add_argument_group("Evaluation")
    eval_group.add_argument("--test-ratio", type=float, default=0.2)
    eval_group.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])

    # --- Output ---
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--model-path", type=str, default="models/als_model.pkl")
    out_group.add_argument("--metrics-path", type=str, default="models/eval_metrics.json")

    return parser.parse_args()


def load_csv_data(path: str) -> pd.DataFrame:
    """
    Load a real interactions CSV and normalize it to the internal schema.

    Expected columns: user_id, item_id, interaction_count
    Returns a DataFrame with columns: user_id, item_id, interaction
    where 'interaction' is clipped to [1, inf) and cast to float32.
    """
    print(f"  Loading data from: {path}")
    df = pd.read_csv(path)

    required = {"user_id", "item_id", "interaction_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df[["user_id", "item_id", "interaction_count"]].copy()
    df.columns = ["user_id", "item_id", "interaction"]
    df["interaction"] = df["interaction"].clip(lower=1).astype("float32")
    df = df.dropna().drop_duplicates(subset=["user_id", "item_id"])
    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)

    print(f"  Loaded {len(df):,} unique interactions | "
          f"{df['user_id'].nunique():,} users | "
          f"{df['item_id'].nunique():,} items")
    return df


def main():
    args = parse_args()

    print("=" * 60)
    print("  RANKING RECOMMENDER SYSTEM — TRAINING PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load or generate data
    # ------------------------------------------------------------------
    if args.data_path:
        print(f"\n[1/4] Loading real data from CSV ...")
        df = load_csv_data(args.data_path)
    else:
        print(f"\n[1/4] Generating synthetic implicit feedback data ...")
        print(f"      Users: {args.num_users} | "
              f"Items: {args.num_items} | "
              f"Interactions: {args.num_interactions}")
        df = generate_implicit_data(
            num_users=args.num_users,
            num_items=args.num_items,
            num_interactions=args.num_interactions,
        )
        print(f"      Unique interactions after dedup: {len(df):,}")

    # ------------------------------------------------------------------
    # Step 2: Train / test split
    # ------------------------------------------------------------------
    print(f"\n[2/4] Splitting data (test_ratio={args.test_ratio}) ...")
    train_df, test_df = train_test_split_implicit(df, test_ratio=args.test_ratio)
    print(f"      Train: {len(train_df):,} interactions | "
          f"Test: {len(test_df):,} interactions")
    print(f"      Train users: {train_df['user_id'].nunique():,} | "
          f"Test users: {test_df['user_id'].nunique():,}")

    # ------------------------------------------------------------------
    # Step 3: Train model
    # ------------------------------------------------------------------
    print(f"\n[3/4] Training ALS model ...")
    print(f"      Factors: {args.factors} | "
          f"Iterations: {args.iterations} | "
          f"Regularization: {args.regularization} | "
          f"Alpha: {args.alpha}")
    model = ImplicitALSModel(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        alpha=args.alpha,
    )
    model.fit(train_df)

    # ------------------------------------------------------------------
    # Step 4: Evaluate with ranking metrics
    # ------------------------------------------------------------------
    print(f"\n[4/4] Evaluating with ranking metrics (K={args.k}) ...")
    metrics = evaluate_model(model, test_df, k_values=args.k)
    print_evaluation_report(metrics)

    # Save model
    model.save(args.model_path)

    # Save metrics
    metrics_dir = os.path.dirname(args.metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)
    with open(args.metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to {args.metrics_path}")

    print("\nTraining pipeline complete. Start the API with:")
    print("  uvicorn api.app:app --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
