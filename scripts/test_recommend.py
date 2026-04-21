"""
Isolated test to debug the recommend() call.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ImplicitALSModel
import scipy.sparse as sp
import numpy as np

model = ImplicitALSModel.load('models/als_model.pkl')

print(f"user_item_matrix shape: {model.user_item_matrix.shape}")
print(f"item_user_matrix shape: {model.item_user_matrix.shape}")
print(f"model.model.user_factors shape: {model.model.user_factors.shape}")
print(f"model.model.item_factors shape: {model.model.item_factors.shape}")
print(f"num users in map: {len(model.user_id_map)}")
print(f"num items in map: {len(model.item_id_map)}")

user_id = 0
user_idx = model.user_id_map[user_id]
print(f"\nuser_id={user_id}, user_idx={user_idx}")

# The user row from user_item_matrix has shape (1, n_items)
user_items_row = model.user_item_matrix[user_idx]
print(f"user_items_row shape: {user_items_row.shape}")
print(f"user_items_row type: {type(user_items_row)}")

# Try the recommend call directly
try:
    item_indices, scores = model.model.recommend(
        user_idx,
        user_items_row,
        N=10,
        filter_already_liked_items=True
    )
    print(f"Success! Top-10 items: {item_indices[:5]}")
except Exception as e:
    print(f"Error: {e}")
    # Try with filter_already_liked_items=False
    try:
        item_indices, scores = model.model.recommend(
            user_idx,
            user_items_row,
            N=10,
            filter_already_liked_items=False
        )
        print(f"Success with filter=False! Top-10 items: {item_indices[:5]}")
    except Exception as e2:
        print(f"Error with filter=False too: {e2}")
