# Film Recommendations

A **movie recommendation system** built for learning and portfolio use. It implements **user-based collaborative filtering** on the [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) dataset, exposes a **REST API** with **FastAPI**, and ships with two frontends: a **static HTML/JS** page and a **React (Vite)** client.

---

## Highlights

- **Collaborative filtering**: user–user cosine similarity on a user–movie rating matrix; top-*N* recommendations with unseen-movie filtering.
- **Cold start**: popular-movie fallback for new users and mean-based fallbacks for sparse cases; configurable minimum rating count for popularity.
- **Offline evaluation**: train/test split, **MAE**, **RMSE**, **Precision@K**, **Recall@K** (ratings ≥ threshold treated as relevant).
- **Production-oriented API**: OpenAPI docs (`/docs`), health check, JSON responses with titles merged from `u.item`.
- **Dual UI**: same backend serves a static demo at `/` and optional React app in `frontend/` for local development.

---

## Tech Stack

| Layer | Technologies |
|--------|----------------|
| Backend | Python 3, **FastAPI**, Uvicorn |
| ML / data | **pandas**, **scikit-learn** (cosine similarity, metrics, train_test_split) |
| Static frontend | HTML, CSS, JavaScript |
| SPA frontend | **React 19**, **Vite** |

---

## Repository Layout

```
.
├── app.py              # FastAPI app: routes, CORS, static mount
├── recommender.py      # Core CF logic, data load, evaluation helpers
├── main.py             # Minimal script baseline (optional)
├── main2.py            # Standalone script with metrics + demo print (optional)
├── u.data              # MovieLens ratings (tab-separated)
├── u.item              # Movie metadata (pipe-separated)
├── static/
│   └── index.html      # Static UI (served at / and /static/...)
└── frontend/           # Vite + React UI (run separately in dev)
    └── src/App.jsx
```

Place **`u.data`** and **`u.item`** in the project root (same directory as `app.py` and `recommender.py`). Paths are resolved relative to `recommender.py` via `pathlib`.

---

## Setup

### 1. Python environment

```bash
cd /path/to/film-rec
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install fastapi "uvicorn[standard]" pandas scikit-learn
```

### 2. Frontend (React only)

```bash
cd frontend
npm install
```

---

## Run the API + Static UI

From the **repository root** (where `u.data` lives):

```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

| URL | Description |
|-----|-------------|
| http://127.0.0.1:8000/ | Static recommendation UI |
| http://127.0.0.1:8000/docs | Swagger / OpenAPI |
| http://127.0.0.1:8000/health | Liveness probe |

---

## Run the React Frontend (development)

The Vite dev server runs on a **different port** (e.g. 5173). Start the API first, then:

```bash
cd frontend
npm run dev
```

By default, `App.jsx` calls the API at `http://127.0.0.1:8000`. Override with a `.env` file:

```env
VITE_API_BASE=http://127.0.0.1:8000
```

Ensure **CORS** is configured on the API for your dev origin (already permissive in `app.py` for local development).

---

## API Overview

### `GET /recommend/{user_id}?top_n=5`

Returns personalized recommendations (movie id, title, score). Validates `top_n` ∈ [1, 50].

**Example response**

```json
{
  "user_id": 1,
  "top_n": 5,
  "items": [
    { "movie_id": 50, "title": "Star Wars (1977)", "score": 4.37 }
  ]
}
```

### `GET /metrics?k=10&rel_threshold=4.0`

Computes offline **MAE**, **RMSE**, **Precision@K**, **Recall@K** on the held-out test split. This endpoint can be slow because it runs full evaluation; use for demos or batch checks, not high-QPS production traffic.

---

## Method Summary

1. Split ratings into **80% train / 20% test** (`random_state=42`).
2. Build a **user × movie** matrix from train ratings; missing entries treated as **0** for similarity (see limitations).
3. **Cosine similarity** between users; for a target user, aggregate scores from top similar neighbors, mask seen movies, take top-*N*; **popular movies** fill shortfalls when needed.
4. **Rating prediction** for metrics uses neighbor means with fallbacks to item/global means.

---

## Limitations & Next Steps

- **Scale**: full user–user similarity is \(O(n_{\text{users}}^2)\) in memory; fine for 100K, not for huge catalogs without factorization or ANN.
- **Matrix sparsity**: zero-filling for similarity is a simplifying choice; production systems often use centered ratings or implicit feedback models.
- **No real-time learning**: matrices are built at import time; updating ratings would require reload or a training pipeline.
- **Possible extensions**: matrix factorization (SVD/ALS), item-based CF, a real database for ratings, authentication, and Docker deployment.

---

## Author

**Qin Hao** — [qh2079382758@gmail.com](mailto:qh2079382758@gmail.com)

Personal / academic portfolio project for internship and job applications.

## License

This project is licensed under the [MIT License](./LICENSE).

Dataset usage follows the [MovieLens terms](https://grouplens.org/datasets/movielens/).
