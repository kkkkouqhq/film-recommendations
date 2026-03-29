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
    return {"status": "ok"}   #健康检查接口，看起来只是个简单又没意义的接口，但可以用来快速判断服务有没有挂

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

    if df.empty:   #.empty判断dataframe是否为空，返回布尔值
        return {"user_id": user_id, "top_n": top_n, "items": []}  #如果推荐结果为空，返回一个空列表，而不是报错

    items = []
    for row in df.itertuples(index=False):
        items.append( #这里是在把dataframe转换为JSON
            {
                "movie_id": int(row.movie_id),   #这里的movie_id和下面的score都是numpy的int和float类型，必须转化为普通的int和float让JSON可以识别
                "title": row.title if pd_notna(row.title) else None, #None表示python里的空，相当于c的null JSON可以识别None但是无法识别NaN
                "score": float(row.score),
            }
        )
    return {"user_id": user_id, "top_n": top_n, "items": items}


def pd_notna(x):
    import pandas as pd    #只在这个函数里import，而不用全局import 优化

    return pd.notna(x)   #判断是否不为NaN的pandas函数，见笔记


@app.get("/metrics")
def metrics(k: int = Query(10, ge=1, le=50), rel_threshold: float = Query(4.0, ge=1.0, le=5.0)):
    """
    计算 MAE / RMSE / Precision@K / Recall@K（较慢，仅用于演示或离线评估）。
    """
    mae, rmse = rec.evaluate_predictions_mae_rmse()
    p_at_k, r_at_k = rec.evaluate_precision_recall_at_k(K=k, rel_threshold=rel_threshold)
    return { #转换为JSON
        "mae": float(mae),
        "rmse": float(rmse),
        f"precision_at_{k}": float(p_at_k),
        f"recall_at_{k}": float(r_at_k),
        "k": k,
        "rel_threshold": rel_threshold,
    }