# Ranking Recommender System

A **production-level recommender system focused on ranking**, built with implicit feedback, Alternating Least Squares (ALS) matrix factorization, and a FastAPI REST interface. The system is evaluated exclusively with ranking metrics (Precision@K, Recall@K, NDCG@K, MAP@K) — no RMSE or rating prediction is used.

---

## Design Principles

| Principle | Implementation |
|---|---|
| **Implicit feedback** | Simulated clicks/interactions (power-law distribution) |
| **Ranking, not prediction** | ALS optimized for ranking; scores used only for ordering |
| **Ranking evaluation** | Precision@K, Recall@K, NDCG@K, MAP@K |
| **Cold-start handling** | Popularity-based fallback for unknown users |
| **Production API** | FastAPI with Pydantic schemas, latency logging, health check |
| **Modularity** | Clean separation across `src/` and `api/` modules |

---

## Project Structure

```
ranking-recommender-system/
├── src/
│   ├── data.py         # Implicit feedback simulation & train/test split
│   ├── model.py        # ALS matrix factorization model
│   ├── recommend.py    # Recommendation engine with cold-start handling
│   └── evaluate.py     # Ranking metrics: Precision@K, Recall@K, NDCG@K, MAP@K
├── api/
│   └── app.py          # FastAPI REST API
├── scripts/
│   └── train.py        # End-to-end training & evaluation pipeline
├── models/             # Saved model artifacts (generated after training)
├── data/               # Data files (generated after training)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python scripts/train.py
```

This will:
- Generate 50,000 implicit interactions across 1,000 users and 5,000 items
- Split into train (80%) and test (20%) sets
- Train an ALS model with 128 latent factors
- Evaluate with Precision@K, Recall@K, NDCG@K, MAP@K at K = 5, 10, 20
- Save the model to `models/als_model.pkl`

**Advanced options:**

```bash
python scripts/train.py \
  --num-users 2000 \
  --num-items 10000 \
  --factors 256 \
  --iterations 100 \
  --alpha 40.0 \
  --k 5 10 20 50
```

### 3. Start the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### `GET /recommend/{user_id}`

Returns top-K ranked item recommendations for a user.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `user_id` | int | required | Target user ID |
| `top_k` | int | 10 | Number of recommendations (1–100) |

**Example Request:**

```bash
curl http://localhost:8000/recommend/42?top_k=10
```

**Example Response:**

```json
{
  "user_id": 42,
  "top_k": 10,
  "cold_start": false,
  "latency_ms": 3.142,
  "recommendations": [
    { "rank": 1, "item_id": 1234, "score": 0.9821, "cold_start": false },
    { "rank": 2, "item_id": 876,  "score": 0.9654, "cold_start": false },
    ...
  ]
}
```

**Cold-start user (unknown user ID):**

```bash
curl http://localhost:8000/recommend/999999
```

Returns popularity-based recommendations with `"cold_start": true`.

---

### `GET /health`

```bash
curl http://localhost:8000/health
# {"status": "ok", "model_loaded": true}
```

### `GET /model/info`

```bash
curl http://localhost:8000/model/info
# {"num_users": 1000, "num_items": 5000, "factors": 128, ...}
```

### `GET /docs`

Interactive Swagger UI available at `http://localhost:8000/docs`.

---

## Python Interface

```python
from src.recommend import recommend

# Known user — ALS-based ranking
recs = recommend(user_id=42, top_k=10)
for r in recs:
    print(f"Rank {r['rank']}: item_id={r['item_id']}, score={r['score']:.4f}")

# Cold-start user — popularity-based fallback
recs = recommend(user_id=999999, top_k=10)
print(recs[0]['cold_start'])  # True
```

---

## Evaluation Metrics

All evaluation is **ranking-based**. The ground truth for each user is the set of items they interacted with in the held-out test set.

| Metric | Definition |
|---|---|
| **Precision@K** | Fraction of top-K recommendations that are relevant |
| **Recall@K** | Fraction of all relevant items that appear in top-K |
| **NDCG@K** | Normalized Discounted Cumulative Gain — rewards relevant items ranked higher |
| **MAP@K** | Mean Average Precision — average precision across all rank positions |

> **No RMSE or rating prediction is used anywhere in this system.**

---

## Cold-Start Strategy

Users not seen during training receive **popularity-based recommendations**: items are ranked by the total number of unique user interactions they received in the training set. This ensures every user — including brand-new ones — receives a meaningful, ranked response.

---

## Model Architecture

The system uses **Alternating Least Squares (ALS)** for implicit feedback (Hu et al., 2008). The interaction matrix is treated as confidence-weighted preferences:

```
c_ui = 1 + α · r_ui
```

where `r_ui` is the raw interaction count and `α` (default: 40) is the confidence scaling factor. ALS alternates between fixing user factors and solving for item factors, and vice versa, minimizing a weighted least-squares objective. The resulting user and item latent vectors are used to rank items by dot-product similarity.

---

## License

MIT
