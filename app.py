# app.py
from fastapi import FastAPI, HTTPException, Query

import recommender as rec

app = FastAPI(
    title="Film Recommender API",
    description="基于用户协同过滤的电影推荐（与 main2 逻辑一致）",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(
    user_id: int,
    top_n: int = Query(5, ge=1, le=50, description="返回前几条推荐"),
):
    """为指定用户返回推荐列表（含片名）。"""
    try:
        df = rec.recommend_with_titles(user_id, top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if df.empty:
        return {"user_id": user_id, "top_n": top_n, "items": []}

    items = []
    for row in df.itertuples(index=False):
        items.append(
            {
                "movie_id": int(row.movie_id),
                "title": row.title if pd_notna(row.title) else None,
                "score": float(row.score),
            }
        )
    return {"user_id": user_id, "top_n": top_n, "items": items}


def pd_notna(x):
    import pandas as pd

    return pd.notna(x)


@app.get("/metrics")
def metrics(k: int = Query(10, ge=1, le=50), rel_threshold: float = Query(4.0, ge=1.0, le=5.0)):
    """
    计算 MAE / RMSE / Precision@K / Recall@K（较慢，仅用于演示或离线评估）。
    """
    mae, rmse = rec.evaluate_predictions_mae_rmse()
    p_at_k, r_at_k = rec.evaluate_precision_recall_at_k(K=k, rel_threshold=rel_threshold)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        f"precision_at_{k}": float(p_at_k),
        f"recall_at_{k}": float(r_at_k),
        "k": k,
        "rel_threshold": rel_threshold,
    }