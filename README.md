# Collaborative Filtering (Hybrid CF + Content-based)

A production-oriented **hybrid recommendation system** combining **item-based collaborative filtering (CF)** and **content-based (CB)** item embeddings.  
This project is built for large-scale e-commerce interaction logs and demonstrates robust engineering patterns such as sparse matrices, TruncatedSVD embeddings, HNSW approximate nearest neighbors, checkpointed evaluation loops, and caching.

---

## 🔗 Dataset

This repo expects the RetailRocket e-commerce dataset (large). Download from Kaggle and place required CSV files in the repository root:

**Kaggle dataset (download manually):**  
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset/data

Required files (place in project root):
- `events.csv` (user interactions: view / addtocart / transaction)
- `item_properties_part1.csv`
- `item_properties_part2.csv`
- `category_tree.csv`

> ⚠️ The dataset is large and cannot be included in the repo. Download from Kaggle, unzip, and move files to the project folder before running.

---

## 🧠 Project Overview

The pipeline implemented in `main()`:

1. **Load & preprocess** `events.csv` (keep `view`, `addtocart`, `transaction`; convert timestamps; map implicit weights).  
2. **Train/test split** by last-N interactions per user (leave-last strategy).  
3. **Build content-based item features** using category properties and the category tree (one-hot + parent expansion).  
4. **Compute CB embeddings** via `TruncatedSVD` on the item-category one-hot matrix and normalize.  
5. **Index embeddings** with **HNSW** (hnswlib) for fast similarity queries.  
6. **Build item-user sparse matrix** (items x users) using interaction weights.  
7. **Item-based CF neighbors**: compute top-K item neighbors using `NearestNeighbors` (cosine) on item rows of the CF matrix, with optional caching.  
8. **Recommendation scoring** per user:
   - CF: aggregate neighbor similarities of user’s interacted items.
   - CB: query HNSW using last item as anchor to get similar items.
   - Hybrid: combine CF & CB via weighted `alpha`.
9. **Evaluation**: precision@K, recall@K, hit rate, NDCG@K, coverage — with checkpointing to resume long runs.  
10. **Outputs**: checkpoint files and final results pickled in `cache_reco/`.

---

## ⚙️ Configuration & Hyperparameters

Located as top-level constants in the code:

- `SVD_DIM = 64` — CB embedding dimension
- `ITEM_NN_TOPK = 200` — precomputed item neighbors
- `CF_NEIGHBORS_TOPK = 50` — neighbors used in scoring
- `HNSW_M = 64`, `HNSW_EF_CONSTRUCTION = 200`, `HNSW_EF_SEARCH = 100` — hnswlib settings
- `MIN_INTERACTIONS_ACTIVE_USER = 1` — min events to keep a user
- `TRAIN_TEST_LAST_N = 3` — last N interactions used as test
- `ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]` — hybrid weights (CF vs CB)
- `K_EVAL = 5` — evaluation cutoff K (Precision@5, etc.)

You can tune these for performance vs. quality tradeoffs.

---

## 📁 Repo / Code structure

collaborative-filtering/  
├── collaborative_filtering.py # or main script (as provided)  
├── cache_reco/ # caches & checkpoints are written here  
├── events.csv # (download from Kaggle) interactions  
├── item_properties_part1.csv # (download from Kaggle)  
├── item_properties_part2.csv # (download from Kaggle)  
├── category_tree.csv # (download from Kaggle)  
├── requirements.txt # Python dependencies (suggested)  
└── README.md

---

## 🚀 How to run

1. Download dataset from Kaggle and place the required CSVs in the repo root.  
2. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
```

```bash
pip install -r requirements.txt
```

3. Run the main script (example filename in repo):

```bash
python collaborative_filtering.py
```

4. Outputs & caches will appear in cache_reco/:  
- item_neighbors.pkl — cached neighbors (if enabled)
- hybrid_eval_checkpoint.pkl — evaluation checkpoint while running
- hybrid_eval_final.pkl — final evaluation results (pickled list / DataFrame)

---

## 📌 Notes on scale & compute

- This pipeline is designed to run on a single powerful machine but is I/O and memory intensive. For large datasets, consider:
    -  running on a machine with enough RAM (or using sparse chunking),
    - using persistent databases for item/user mappings,
    - offloading heavy nearest neighbor computation to approximate indexes (HNSW is already used), or
    - distributed frameworks (Spark) if needed.
- Caching (pickle files) is implemented to avoid recomputing heavy neighbors and embeddings.

---

## 🧪 Evaluation

Evaluation is performed per ALPHAS values and computes:
- Precision@K, Recall@K, HitRate@K, NDCG@K, Coverage (unique recommended items / total items).
Checkpointing saves intermediate state to cache_reco/hybrid_eval_checkpoint.pkl so long runs can be resumed from the last saved user index / alpha.

---

## ✅ Reproducibility tips

- Persist random seeds where applicable (TruncatedSVD uses random_state=42 in code).
- Ensure identical item & user indexing across runs — the script performs several synchronization steps to align CF and CB item domains.
- Use the provided caching mechanism to speed up iterative experiments (neighbors, embedding index).

---

## ⚠️ Legal & Ethical Notice

This project processes user interaction logs for research and recommendation experiments. Use data responsibly and follow applicable privacy laws and dataset licensing terms. Do not publish personal or sensitive user information.

---

## 🛠️ Potential Improvements / Roadmap

- Move heavy computations to GPU or distributed compute (FAISS GPU, Spark).
- Add streaming updates to HNSW / incremental neighbor updates.
- Use matrix factorization (ALS) as additional baseline.
- Add more robust evaluation splits (time-aware cross-validation).
- Export recommendation API (FastAPI/Flask) and dashboard (Streamlit) for demo.

---

## 👤 Author

Ch1ns0n

Machine Learning Engineer | Data Engineer

🔗 [GitHub](https://github.com/ch1ns0n)  
💼 [LinkedIn](https://www.linkedin.com/in/samuelchinson)