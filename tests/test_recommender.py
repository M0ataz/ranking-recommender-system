import pytest
import os
import sys
from unittest.mock import patch

# Add project root to sys.path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommend import recommend
from src.data import generate_implicit_data, train_test_split_implicit
from src.model import ImplicitALSModel

@pytest.fixture(scope="module")
def train_synthetic_model():
    """
    Train a model with synthetic data (100 users, 500 items) for testing.
    Saves it to a temporary location.
    """
    # Use small synthetic data
    df = generate_implicit_data(num_users=100, num_items=500, num_interactions=1000)
    train_df, _ = train_test_split_implicit(df, test_ratio=0.1)
    
    model = ImplicitALSModel(factors=16, iterations=5)
    model.fit(train_df)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/test_als_model.pkl"
    model.save(model_path)
    
    yield model_path
    
    # Cleanup after tests
    if os.path.exists(model_path):
        os.remove(model_path)

def test_recommend_known_user(train_synthetic_model):
    """
    Test recommend for a known user (user_id=0).
    Assert len == 5.
    """
    model_path = train_synthetic_model
    # Clear the global engine to ensure we load the test model
    import src.recommend
    src.recommend._engine = None
    
    recs = recommend(user_id=0, top_k=5, model_path=model_path)
    assert len(recs) == 5

def test_recommend_returns_ranks(train_synthetic_model):
    """
    Test that the returned recommendations have a 'rank' field starting at 1.
    """
    model_path = train_synthetic_model
    import src.recommend
    src.recommend._engine = None
    
    recs = recommend(user_id=0, top_k=5, model_path=model_path)
    assert len(recs) > 0
    assert recs[0]['rank'] == 1

def test_cold_start(train_synthetic_model):
    """
    Test recommend for an unknown user (user_id=99999).
    Assert cold_start == True.
    """
    model_path = train_synthetic_model
    import src.recommend
    src.recommend._engine = None
    
    recs = recommend(user_id=99999, top_k=5, model_path=model_path)
    assert len(recs) > 0
    assert recs[0]['cold_start'] is True

def test_scores_ordered(train_synthetic_model):
    """
    Test that the scores in the recommendations are in descending order.
    """
    model_path = train_synthetic_model
    import src.recommend
    src.recommend._engine = None
    
    recs = recommend(user_id=0, top_k=5, model_path=model_path)
    assert len(recs) > 1
    for i in range(len(recs) - 1):
        assert recs[i]['score'] >= recs[i+1]['score']
