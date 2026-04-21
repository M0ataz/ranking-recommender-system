import os
import sys
from typing import List, Dict, Any, Optional

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ImplicitALSModel


class RecommendationEngine:
    """
    High-level recommendation interface that wraps the ALS model.

    Provides:
    - recommend(user_id): Returns top-K ranked items with scores.
    - Cold-start handling: Falls back to popularity-based ranking for unknown users.
    - Lazy model loading: Loads the model from disk on first use.
    """

    def __init__(self, model_path: str = "models/als_model.pkl"):
        self.model_path = model_path
        self._model: Optional[ImplicitALSModel] = None

    def _load_model(self) -> ImplicitALSModel:
        """Lazy-load the model from disk."""
        if self._model is None:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at '{self.model_path}'. "
                    "Run `python scripts/train.py` to train and save the model first."
                )
            self._model = ImplicitALSModel.load(self.model_path)
        return self._model

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        filter_already_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Return top-K ranked item recommendations for a given user.

        Handles two scenarios:
        - **Known user**: Uses ALS latent factors to rank all items by predicted
          relevance score and returns the top-K unseen items.
        - **Cold-start (unknown user)**: Falls back to a popularity-based ranking
          of the most interacted items across all users.

        Args:
            user_id: The user's ID. Can be a known or unknown (cold-start) user.
            top_k: Number of recommendations to return. Defaults to 10.
            filter_already_interacted: If True, exclude items the user has already
                interacted with (only applies to known users).

        Returns:
            A list of dicts, each containing:
                - rank (int): 1-indexed position in the ranked list.
                - item_id (int): The recommended item's ID.
                - score (float): The relevance score (higher = more relevant).
                - cold_start (bool): Whether the recommendation was generated via
                  cold-start fallback.
        """
        model = self._load_model()

        is_cold_start = user_id not in model.user_id_map

        ranked_items = model.rank_items(
            user_id=user_id,
            top_k=top_k,
            filter_already_interacted=filter_already_interacted
        )

        results = []
        for rank, (item_id, score) in enumerate(ranked_items, start=1):
            results.append({
                "rank": rank,
                "item_id": item_id,
                "score": round(score, 6),
                "cold_start": is_cold_start
            })

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        model = self._load_model()
        return {
            "num_users": len(model.user_id_map),
            "num_items": len(model.item_id_map),
            "factors": model.factors,
            "regularization": model.regularization,
            "iterations": model.iterations,
            "alpha": model.alpha,
        }


# Module-level singleton for use by the API layer
_engine: Optional[RecommendationEngine] = None


def get_engine(model_path: str = "models/als_model.pkl") -> RecommendationEngine:
    """Return the global RecommendationEngine singleton."""
    global _engine
    if _engine is None:
        _engine = RecommendationEngine(model_path=model_path)
    return _engine


def recommend(
    user_id: int,
    top_k: int = 10,
    model_path: str = "models/als_model.pkl"
) -> List[Dict[str, Any]]:
    """
    Module-level convenience function: recommend(user_id).

    This is the primary system interface as specified in the project requirements.

    Args:
        user_id: The target user's ID.
        top_k: Number of top-K ranked items to return.
        model_path: Path to the saved model file.

    Returns:
        List of ranked recommendation dicts with keys:
        rank, item_id, score, cold_start.

    Example:
        >>> from src.recommend import recommend
        >>> recs = recommend(user_id=42, top_k=10)
        >>> for r in recs:
        ...     print(r['rank'], r['item_id'], r['score'])
    """
    engine = get_engine(model_path=model_path)
    return engine.recommend(user_id=user_id, top_k=top_k)


if __name__ == "__main__":
    print("=== RecommendationEngine Demo ===\n")

    engine = RecommendationEngine(model_path="models/als_model.pkl")

    # Test with a known user (assumes model is trained)
    try:
        print("Known user recommendations (user_id=0):")
        recs = engine.recommend(user_id=0, top_k=10)
        for r in recs:
            print(f"  Rank {r['rank']:2d} | item_id={r['item_id']:5d} | score={r['score']:.6f} | cold_start={r['cold_start']}")

        print("\nCold-start user recommendations (user_id=999999):")
        cold_recs = engine.recommend(user_id=999999, top_k=10)
        for r in cold_recs:
            print(f"  Rank {r['rank']:2d} | item_id={r['item_id']:5d} | score={r['score']:.6f} | cold_start={r['cold_start']}")

        print("\nModel Info:")
        info = engine.get_model_info()
        for k, v in info.items():
            print(f"  {k}: {v}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run `python scripts/train.py` first to train the model.")
