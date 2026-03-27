import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 读取数据
ratings = pd.read_csv(
    "u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "u.item",
    sep="|",
    encoding="latin-1",
    usecols=[0, 1],
    names=["movie_id", "title"]
)

# 构建矩阵
user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
).fillna(0)

# 计算相似度
user_similarity = cosine_similarity(user_movie_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

# 推荐函数
def recommend_movies(user_id, top_n=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    similar_users = similar_users.drop(user_id)

    top_users = similar_users.head(10).index

    similar_users_movies = user_movie_matrix.loc[top_users]

    movie_scores = similar_users_movies.mean(axis=0)

    user_movies = user_movie_matrix.loc[user_id]

    unseen_movies = user_movies[user_movies == 0]

    recommendations = movie_scores[unseen_movies.index]

    recommendations = recommendations.sort_values(ascending=False)

    recommendations = recommendations.head(top_n)

    # 加电影名
    recommendations = recommendations.reset_index()
    recommendations.columns = ["movie_id", "score"]

    recommendations = recommendations.merge(movies, on="movie_id")

    return recommendations