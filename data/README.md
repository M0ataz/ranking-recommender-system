# Data Directory

This directory contains the datasets used for training and evaluating the ranking recommender system.

## Current Synthetic Data Schema

By default, the system generates synthetic implicit feedback data using a power-law distribution to simulate realistic user activity and item popularity. The generated data is split into `train.csv` and `test.csv`.

The schema for the synthetic data is as follows:

| Column | Type | Description |
|---|---|---|
| `user_id` | integer | Unique identifier for the user |
| `item_id` | integer | Unique identifier for the item |
| `interaction` | integer | Binary indicator (1) representing an implicit interaction (e.g., click, view) |

## Using Real Data (MovieLens 100K)

You can replace the synthetic data with real-world datasets like MovieLens 100K. To do this, the system requires a CSV file with the following columns:

| Column | Type | Description |
|---|---|---|
| `user_id` | integer | Unique identifier for the user |
| `item_id` | integer | Unique identifier for the item |
| `interaction_count` | integer | The number of times the user interacted with the item, or a rating value treated as an interaction weight |

### How to Download and Use MovieLens 100K

1. **Download and Parse:**
   Run the provided script to download the MovieLens 100K dataset and parse it into the required format.
   ```bash
   python scripts/download_movielens.py
   ```
   This script downloads the dataset from [GroupLens](https://files.grouplens.org/datasets/movielens/ml-100k.zip), extracts the `u.data` file, and saves it as `data/interactions.csv` with the required `user_id`, `item_id`, and `interaction_count` columns.

2. **Train with Real Data:**
   Update your training command to point to the newly generated CSV file using the `--data-path` argument.
   ```bash
   python scripts/train.py --data-path data/interactions.csv
   ```

By following these steps, the system will automatically split the real data into training and testing sets and train the ALS model using the actual user-item interactions.
