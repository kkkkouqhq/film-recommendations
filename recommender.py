# recommender.py — 电影推荐核心逻辑（供脚本或 FastAPI 导入）
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

# 数据路径（相对本文件所在目录）
RATINGS_PATH = BASE_DIR / "u.data"
MOVIES_PATH = BASE_DIR / "u.item"

ratings = pd.read_csv(
    RATINGS_PATH,
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

user_movie_matrix = train_df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating",
)
user_movie_matrix = user_movie_matrix.fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index,
)

global_mean = float(train_df["rating"].mean())

movie_stats = train_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()
min_count = 20
popular_movies = movie_stats[movie_stats["count"] >= min_count].sort_values(
    ["mean", "count"], ascending=[False, False]
)

popular_scores_series = popular_movies.set_index("movie_id")["mean"].sort_values(ascending=False)

if len(popular_scores_series) == 0:
    popular_scores_series = pd.Series(dtype=float)

movies = pd.read_csv(
    MOVIES_PATH,
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"],
)


def recommend_movies(user_id, user_movie_matrix, user_similarity_df, top_n=5, top_users_n=10):
    if user_id not in user_movie_matrix.index:
        if len(popular_scores_series) >= top_n:
            return popular_scores_series.head(top_n)
        out = popular_scores_series.copy()
        need = top_n - len(out)
        if need > 0:
            out = pd.concat(
                [out, pd.Series([global_mean] * need, index=[])],
            ).sort_values(ascending=False)
        return out.head(top_n)

    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)
    top_users = similar_users.head(top_users_n).index

    similar_users_movies = user_movie_matrix.loc[top_users]
    movie_scores = similar_users_movies.mean(axis=0)

    user_movies = user_movie_matrix.loc[user_id]
    unseen_movie_ids = user_movies[user_movies == 0].index

    recommendations = movie_scores.loc[unseen_movie_ids]
    recommendations = recommendations[recommendations > 0]
    recommendations = recommendations.sort_values(ascending=False).head(top_n)

    if len(recommendations) < top_n and len(popular_scores_series) > 0:
        missing = top_n - len(recommendations)
        already = set(recommendations.index.tolist())
        popular_to_add = popular_scores_series[
            ~popular_scores_series.index.isin(already)
        ].head(missing)
        recommendations = pd.concat([recommendations, popular_to_add], axis=0).sort_values(
            ascending=False
        ).head(top_n)

    return recommendations


def predict_rating(user_id, movie_id, user_movie_matrix, user_similarity_df, top_users_n=10):
    if user_id not in user_movie_matrix.index:
        return global_mean

    if movie_id not in user_movie_matrix.columns:
        return global_mean

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    top_users = similar_users.head(top_users_n).index

    neighbor_ratings = user_movie_matrix.loc[top_users, movie_id]
    valid = neighbor_ratings[neighbor_ratings > 0]
    if len(valid) == 0:
        movie_nonzero = user_movie_matrix[movie_id]
        movie_nonzero = movie_nonzero[movie_nonzero > 0]
        if len(movie_nonzero) > 0:
            return float(movie_nonzero.mean())
        return global_mean

    return float(valid.mean())


def evaluate_predictions_mae_rmse():
    y_true = []
    y_pred = []
    for row in test_df.itertuples(index=False):
        pred = predict_rating(
            row.user_id,
            row.movie_id,
            user_movie_matrix,
            user_similarity_df,
        )
        y_true.append(row.rating)
        y_pred.append(pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return mae, rmse


def evaluate_precision_recall_at_k(K=10, rel_threshold=4.0):
    precisions = []
    recalls = []
    for uid, user_test in test_df.groupby("user_id"):
        relevant = set(user_test[user_test["rating"] >= rel_threshold]["movie_id"].tolist())
        if len(relevant) == 0:
            continue
        rec_series = recommend_movies(
            uid,
            user_movie_matrix,
            user_similarity_df,
            top_n=K,
        )
        recommended = set(rec_series.index.tolist())
        hits = len(recommended & relevant)
        precisions.append(hits / K)
        recalls.append(hits / len(relevant))
    if len(precisions) == 0:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))


def recommend_with_titles(user_id: int, top_n: int = 5):
    """返回带片名的 DataFrame：movie_id, title, score"""
    rec = recommend_movies(user_id, user_movie_matrix, user_similarity_df, top_n=top_n)
    rec = rec.reset_index()
    rec.columns = ["movie_id", "score"]
    return rec.merge(movies, on="movie_id", how="left")