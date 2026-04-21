"""
Diagnostic script to check data density and model sanity.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data import generate_implicit_data, train_test_split_implicit
from src.model import ImplicitALSModel

df = generate_implicit_data()
train_df, test_df = train_test_split_implicit(df)

print(f"Total interactions: {len(df)}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Unique items: {df['item_id'].nunique()}")
print(f"Avg interactions per user: {df.groupby('user_id').size().mean():.1f}")
print(f"Avg interactions per item: {df.groupby('item_id').size().mean():.1f}")
print(f"Sparsity: {1 - len(df)/(df['user_id'].nunique()*df['item_id'].nunique()):.4%}")

# Check test set
test_per_user = test_df.groupby('user_id').size()
print(f"\nTest set:")
print(f"  Users with test items: {len(test_per_user)}")
print(f"  Avg test items per user: {test_per_user.mean():.1f}")
print(f"  Min/Max test items: {test_per_user.min()} / {test_per_user.max()}")

# Train a quick model and check if it can recall test items
model = ImplicitALSModel(factors=64, iterations=20)
model.fit(train_df)

# Check a specific user
sample_user = test_df['user_id'].iloc[0]
test_items = set(test_df[test_df['user_id'] == sample_user]['item_id'])
print(f"\nSample user {sample_user}: {len(test_items)} test items")

# Get top-50 recs without filtering
import scipy.sparse as sp
user_idx = model.user_id_map[sample_user]
n_items = len(model.item_id_map)
empty_row = sp.csr_matrix((1, n_items), dtype=np.float32)
item_indices, scores = model.model.recommend(user_idx, empty_row, N=50, filter_already_liked_items=False)
rec_ids = [model.item_idx_map[int(i)] for i in item_indices]

hits = set(rec_ids) & test_items
print(f"Hits in top-50: {len(hits)} / {len(test_items)}")
print(f"Top-10 scores: {scores[:10]}")
