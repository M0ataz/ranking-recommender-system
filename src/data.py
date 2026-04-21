import numpy as np
import pandas as pd
from typing import Tuple

def generate_implicit_data(
    num_users: int = 1000,
    num_items: int = 5000,
    num_interactions: int = 50000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate simulated implicit feedback data (e.g., clicks, views, purchases).
    Simulates realistic behavior where some items are more popular (power-law distribution)
    and some users are more active.
    """
    np.random.seed(random_seed)
    
    # Generate user activity levels (power-law)
    user_activity = np.random.power(2.5, num_users)
    user_probs = user_activity / user_activity.sum()
    
    # Generate item popularity levels (power-law)
    item_popularity = np.random.power(1.5, num_items)
    item_probs = item_popularity / item_popularity.sum()
    
    # Sample interactions
    users = np.random.choice(np.arange(num_users), size=num_interactions, p=user_probs)
    items = np.random.choice(np.arange(num_items), size=num_interactions, p=item_probs)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': users,
        'item_id': items,
        'interaction': 1  # Implicit feedback is binary (interacted or not)
    })
    
    # Drop duplicates (if a user interacted with an item multiple times, we just count it as 1 interaction for basic implicit feedback)
    df = df.drop_duplicates(subset=['user_id', 'item_id'])
    
    return df

def train_test_split_implicit(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split implicit feedback data into train and test sets.
    Ensures that users in the test set have at least some interactions in the train set
    to allow for meaningful evaluation (except for cold-start testing).
    """
    np.random.seed(random_seed)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Group by user
    grouped = df.groupby('user_id')
    
    train_list = []
    test_list = []
    
    for user_id, group in grouped:
        n_interactions = len(group)
        if n_interactions >= 5:
            # If user has enough interactions, split them
            n_test = max(1, int(n_interactions * test_ratio))
            test_list.append(group.iloc[:n_test])
            train_list.append(group.iloc[n_test:])
        else:
            # If user has very few interactions, keep all in train to avoid cold-start in test
            # (We will handle cold-start separately)
            train_list.append(group)
            
    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame(columns=df.columns)
    
    return train_df, test_df

if __name__ == "__main__":
    print("Generating data...")
    df = generate_implicit_data()
    print(f"Generated {len(df)} unique interactions.")
    
    train_df, test_df = train_test_split_implicit(df)
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Save to data directory
    import os
    os.makedirs('../data', exist_ok=True)
    train_df.to_csv('../data/train.csv', index=False)
    test_df.to_csv('../data/test.csv', index=False)
    print("Data saved to ../data/")
