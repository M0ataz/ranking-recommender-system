"""
Training Script
---------------
Generates implicit feedback data, trains the ALS model,
evaluates it with ranking metrics, and saves the model to disk.

Usage:
    python scripts/train.py [--factors 128] [--iterations 50] [--k 10 20]
"""

import argparse
import os
import sys
import json

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data import generate_implicit_data, train_test_split_implicit
from src.model import ImplicitALSModel
from src.evaluate import evaluate_model, print_evaluation_report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the ranking recommender system.")
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--num-items", type=int, default=5000)
    parser.add_argument("--num-interactions", type=int, default=50000)
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--model-path", type=str, default="models/als_model.pkl")
    parser.add_argument("--metrics-path", type=str, default="models/eval_metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  RANKING RECOMMENDER SYSTEM — TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Generate data
    print(f"\n[1/4] Generating implicit feedback data...")
    print(f"      Users: {args.num_users}, Items: {args.num_items}, Interactions: {args.num_interactions}")
    df = generate_implicit_data(
        num_users=args.num_users,
        num_items=args.num_items,
        num_interactions=args.num_interactions
    )
    print(f"      Unique interactions after dedup: {len(df)}")

    # Step 2: Train/test split
    print(f"\n[2/4] Splitting data (test ratio={args.test_ratio})...")
    train_df, test_df = train_test_split_implicit(df, test_ratio=args.test_ratio)
    print(f"      Train: {len(train_df)} interactions | Test: {len(test_df)} interactions")
    print(f"      Train users: {train_df['user_id'].nunique()} | Test users: {test_df['user_id'].nunique()}")

    # Step 3: Train model
    print(f"\n[3/4] Training ALS model...")
    print(f"      Factors: {args.factors}, Iterations: {args.iterations}, "
          f"Regularization: {args.regularization}, Alpha: {args.alpha}")
    model = ImplicitALSModel(
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        alpha=args.alpha
    )
    model.fit(train_df)

    # Step 4: Evaluate with ranking metrics
    print(f"\n[4/4] Evaluating with ranking metrics (K={args.k})...")
    metrics = evaluate_model(model, test_df, k_values=args.k)
    print_evaluation_report(metrics)

    # Save model
    model.save(args.model_path)

    # Save metrics
    os.makedirs(os.path.dirname(args.metrics_path) if os.path.dirname(args.metrics_path) else '.', exist_ok=True)
    with open(args.metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Evaluation metrics saved to {args.metrics_path}")

    print("\nTraining pipeline complete. You can now start the API:")
    print("  uvicorn api.app:app --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
