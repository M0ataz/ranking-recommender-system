import numpy as np
import pandas as pd
from typing import Dict, List, Set
from src.model import ImplicitALSModel


def precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Precision@K: Fraction of top-K recommended items that are relevant.

    P@K = |{recommended[:K]} ∩ {relevant}| / K
    """
    if k == 0:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Recall@K: Fraction of all relevant items that appear in the top-K recommendations.

    R@K = |{recommended[:K]} ∩ {relevant}| / |{relevant}|
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = len(set(top_k) & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K (NDCG@K).
    Measures ranking quality, rewarding relevant items appearing earlier.
    """
    top_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(top_k)
        if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended: List[int], relevant: Set[int], k: int) -> float:
    """
    Average Precision @ K (AP@K).
    Computes the average of precision values at each rank where a hit occurs.
    """
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = 0
    precision_sum = 0.0
    for i, item in enumerate(top_k):
        if item in relevant:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / min(len(relevant), k)


def evaluate_model(
    model: ImplicitALSModel,
    test_df: pd.DataFrame,
    k_values: List[int] = [5, 10, 20],
    sample_users: int = 500
) -> Dict[str, float]:
    """
    Evaluate the ranking model on the test set using ranking metrics.

    Evaluation protocol:
    - Ground truth for each user = items they interacted with in the test set.
    - We rank ALL items WITHOUT filtering train interactions, so that test items
      can appear in the ranked list. This is the standard leave-one-out / held-out
      evaluation protocol for implicit feedback recommenders.
    - No RMSE or rating prediction is used anywhere.

    Args:
        model: Fitted ImplicitALSModel instance.
        test_df: Test DataFrame with columns ['user_id', 'item_id', 'interaction'].
        k_values: List of K values to evaluate at.
        sample_users: Number of users to evaluate (for speed). Use -1 for all.

    Returns:
        Dict of metric names to mean values across all evaluated users.
    """
    # Build ground truth: user -> set of relevant items in test set
    test_relevant = (
        test_df.groupby('user_id')['item_id']
        .apply(set)
        .to_dict()
    )

    # Only evaluate users who are in the model (known users)
    known_test_users = [u for u in test_relevant if u in model.user_id_map]

    if sample_users > 0 and len(known_test_users) > sample_users:
        np.random.seed(42)
        known_test_users = list(np.random.choice(known_test_users, size=sample_users, replace=False))

    print(f"Evaluating on {len(known_test_users)} users...")

    max_k = max(k_values)
    metrics_accum = {f"precision@{k}": [] for k in k_values}
    metrics_accum.update({f"recall@{k}": [] for k in k_values})
    metrics_accum.update({f"ndcg@{k}": [] for k in k_values})
    metrics_accum.update({f"map@{k}": [] for k in k_values})

    for user_id in known_test_users:
        relevant_items = test_relevant[user_id]
        if not relevant_items:
            continue

        # Map relevant items to known item indices (skip items not in train vocab)
        relevant_in_vocab = {iid for iid in relevant_items if iid in model.item_id_map}
        if not relevant_in_vocab:
            continue

        # Rank ALL items (no filtering) so test items can appear in recommendations
        ranked = model.rank_items(user_id, top_k=max_k, filter_already_interacted=False)
        recommended_ids = [item_id for item_id, _ in ranked]

        for k in k_values:
            metrics_accum[f"precision@{k}"].append(
                precision_at_k(recommended_ids, relevant_in_vocab, k)
            )
            metrics_accum[f"recall@{k}"].append(
                recall_at_k(recommended_ids, relevant_in_vocab, k)
            )
            metrics_accum[f"ndcg@{k}"].append(
                ndcg_at_k(recommended_ids, relevant_in_vocab, k)
            )
            metrics_accum[f"map@{k}"].append(
                average_precision_at_k(recommended_ids, relevant_in_vocab, k)
            )

    # Average across users
    results = {
        metric: float(np.mean(values))
        for metric, values in metrics_accum.items()
        if values
    }
    return results


def print_evaluation_report(metrics: Dict[str, float]) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 60)
    print("  RANKING EVALUATION REPORT")
    print("=" * 60)
    print(f"  {'Metric':<20} {'Value':>10}")
    print("-" * 60)
    for metric, value in sorted(metrics.items()):
        print(f"  {metric:<20} {value:>10.4f}")
    print("=" * 60)
    print("  NOTE: All metrics are ranking-based (no RMSE used).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data import generate_implicit_data, train_test_split_implicit

    print("Generating data...")
    df = generate_implicit_data()
    train_df, test_df = train_test_split_implicit(df)

    print("Training model...")
    model = ImplicitALSModel(factors=64, iterations=20)
    model.fit(train_df)

    print("Running evaluation...")
    metrics = evaluate_model(model, test_df, k_values=[5, 10, 20])
    print_evaluation_report(metrics)
