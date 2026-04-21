import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import os
from typing import Dict, List, Tuple, Optional
import implicit
from implicit.als import AlternatingLeastSquares


class ImplicitALSModel:
    """
    Matrix Factorization model using Alternating Least Squares (ALS)
    optimized for implicit feedback ranking.

    ALS for implicit feedback (Hu et al., 2008) treats the interaction
    matrix as confidence-weighted preferences rather than explicit ratings.
    The model learns latent user/item factors to rank items by relevance.

    NOTE on implicit library orientation:
    - `model.fit(item_user_matrix)` expects shape (n_items, n_users)
    - After fitting: model.user_factors has shape (n_items, factors)
                     model.item_factors has shape (n_users, factors)
    - This naming is confusing; we use manual dot-product scoring to avoid
      the library's internal orientation assumptions.
    """

    def __init__(
        self,
        factors: int = 128,
        regularization: float = 0.01,
        iterations: int = 50,
        alpha: float = 40.0,
        random_seed: int = 42
    ):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.random_seed = random_seed

        self.model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=random_seed,
            use_gpu=False
        )

        # Mappings from raw IDs to matrix indices
        self.user_id_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.user_idx_map: Dict[int, int] = {}  # index -> user_id
        self.item_idx_map: Dict[int, int] = {}  # index -> item_id

        # Latent factor matrices (set after fit)
        # user_factors: shape (n_users, factors)
        # item_factors: shape (n_items, factors)
        self.user_factors: Optional[np.ndarray] = None
        self.item_factors: Optional[np.ndarray] = None

        # Sparse user-item matrix for filtering already-seen items
        self.user_item_matrix: Optional[sp.csr_matrix] = None

        # Popularity scores for cold-start fallback
        self.item_popularity: Optional[np.ndarray] = None
        self.popular_items: Optional[List[int]] = None

        self._is_fitted = False

    def _build_mappings(self, df: pd.DataFrame) -> None:
        """Build bidirectional mappings between raw IDs and matrix indices."""
        unique_users = sorted(df['user_id'].unique())
        unique_items = sorted(df['item_id'].unique())

        self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
        self.user_idx_map = {idx: uid for uid, idx in self.user_id_map.items()}
        self.item_idx_map = {idx: iid for iid, idx in self.item_id_map.items()}

    def _build_interaction_matrix(self, df: pd.DataFrame) -> sp.csr_matrix:
        """
        Build a sparse user-item interaction matrix with confidence weights.
        Confidence c_ui = 1 + alpha * r_ui where r_ui is the raw interaction count.
        """
        user_indices = df['user_id'].map(self.user_id_map).values
        item_indices = df['item_id'].map(self.item_id_map).values
        interactions = df['interaction'].values.astype(np.float32)
        confidence = 1.0 + self.alpha * interactions

        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)

        matrix = sp.csr_matrix(
            (confidence, (user_indices, item_indices)),
            shape=(n_users, n_items)
        )
        return matrix

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit the ALS model on training implicit feedback data.

        Args:
            train_df: DataFrame with columns ['user_id', 'item_id', 'interaction'].
        """
        print("Building ID mappings...")
        self._build_mappings(train_df)

        print("Building sparse interaction matrix...")
        self.user_item_matrix = self._build_interaction_matrix(train_df)

        # Compute item popularity for cold-start fallback
        item_interaction_counts = np.asarray(self.user_item_matrix.sum(axis=0)).flatten()
        self.item_popularity = item_interaction_counts
        sorted_item_indices = np.argsort(item_interaction_counts)[::-1]
        self.popular_items = [self.item_idx_map[idx] for idx in sorted_item_indices]

        n_users, n_items = self.user_item_matrix.shape
        print(f"Training ALS model on {n_users} users x {n_items} items...")

        # implicit.fit() expects item-user matrix: shape (n_items, n_users)
        item_user_matrix = self.user_item_matrix.T.tocsr()
        self.model.fit(item_user_matrix)

        # After fit on item_user_matrix (n_items, n_users):
        #   model.user_factors -> shape (n_items, factors)  [these are ITEM embeddings]
        #   model.item_factors -> shape (n_users, factors)  [these are USER embeddings]
        # We extract them with correct semantic names:
        self.item_factors = self.model.user_factors  # shape: (n_items, factors)
        self.user_factors = self.model.item_factors  # shape: (n_users, factors)

        self._is_fitted = True
        print("Model training complete.")

    def rank_items(
        self,
        user_id: int,
        top_k: int = 10,
        filter_already_interacted: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Rank items for a given user and return top-K with scores.

        Uses manual dot-product scoring: score(u, i) = user_factors[u] · item_factors[i]
        This avoids the confusing orientation of the implicit library's recommend() method.

        Args:
            user_id: Raw user ID.
            top_k: Number of top items to return.
            filter_already_interacted: Whether to exclude already-seen items.

        Returns:
            List of (item_id, score) tuples sorted by score descending.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling rank_items().")

        if user_id not in self.user_id_map:
            # Cold-start: user not seen during training — return popular items
            return [
                (item_id, float(self.item_popularity[self.item_id_map[item_id]])
                 if item_id in self.item_id_map else 0.0)
                for item_id in self.popular_items[:top_k]
            ]

        user_idx = self.user_id_map[user_id]

        # Compute scores for all items via dot product: (n_items,)
        user_vec = self.user_factors[user_idx]          # shape: (factors,)
        scores = self.item_factors.dot(user_vec)        # shape: (n_items,)

        if filter_already_interacted:
            # Zero out scores for items the user has already interacted with
            seen_item_indices = self.user_item_matrix[user_idx].indices
            scores[seen_item_indices] = -np.inf

        # Get top-K item indices by score (descending)
        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            # argpartition is O(n) vs argsort O(n log n) — faster for large item sets
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for item_idx in top_indices:
            raw_item_id = self.item_idx_map[int(item_idx)]
            results.append((raw_item_id, float(scores[item_idx])))

        return results

    def save(self, path: str) -> None:
        """Persist the trained model and mappings to disk."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'user_factors': self.user_factors,
                'item_factors': self.item_factors,
                'user_id_map': self.user_id_map,
                'item_id_map': self.item_id_map,
                'user_idx_map': self.user_idx_map,
                'item_idx_map': self.item_idx_map,
                'user_item_matrix': self.user_item_matrix,
                'item_popularity': self.item_popularity,
                'popular_items': self.popular_items,
                'factors': self.factors,
                'regularization': self.regularization,
                'iterations': self.iterations,
                'alpha': self.alpha,
            }, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'ImplicitALSModel':
        """Load a previously saved model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        instance = cls(
            factors=data['factors'],
            regularization=data['regularization'],
            iterations=data['iterations'],
            alpha=data['alpha']
        )
        instance.user_factors = data['user_factors']
        instance.item_factors = data['item_factors']
        instance.user_id_map = data['user_id_map']
        instance.item_id_map = data['item_id_map']
        instance.user_idx_map = data['user_idx_map']
        instance.item_idx_map = data['item_idx_map']
        instance.user_item_matrix = data['user_item_matrix']
        instance.item_popularity = data['item_popularity']
        instance.popular_items = data['popular_items']
        instance._is_fitted = True
        print(f"Model loaded from {path}")
        return instance


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data import generate_implicit_data, train_test_split_implicit

    df = generate_implicit_data()
    train_df, test_df = train_test_split_implicit(df)

    model = ImplicitALSModel(factors=64, iterations=20)
    model.fit(train_df)

    sample_user = train_df['user_id'].iloc[0]
    recommendations = model.rank_items(sample_user, top_k=10)
    print(f"\nTop-10 recommendations for user {sample_user}:")
    for rank, (item_id, score) in enumerate(recommendations, 1):
        print(f"  Rank {rank}: item_id={item_id}, score={score:.4f}")

    model.save('models/als_model.pkl')
