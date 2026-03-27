import pandas as pd

# 读取数据
ratings = pd.read_csv(
    "u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

# 构建用户-电影矩阵
user_movie_matrix = ratings.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
)

# 填充0
user_movie_matrix = user_movie_matrix.fillna(0)

from sklearn.metrics.pairwise import cosine_similarity

# 计算用户相似度
user_similarity = cosine_similarity(user_movie_matrix)

# 转成DataFrame方便看
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)

def recommend_movies(user_id, user_movie_matrix, user_similarity_df, top_n=5):
    # 1. 找到和该用户相似的用户
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # 去掉自己
    similar_users = similar_users.drop(user_id)

    # 2. 找最相似的用户（取前10个）
    top_users = similar_users.head(10).index

    # 3. 这些用户喜欢的电影（评分高的）
    similar_users_movies = user_movie_matrix.loc[top_users]

    # 4. 计算平均评分
    movie_scores = similar_users_movies.mean(axis=0)

    # 5. 去掉该用户已经看过的电影
    user_movies = user_movie_matrix.loc[user_id]
    unseen_movies = user_movies[user_movies == 0]

    # 6. 只保留没看过的电影
    recommendations = movie_scores[unseen_movies.index]

    # 7. 排序
    recommendations = recommendations.sort_values(ascending=False)

    return recommendations.head(top_n)


# 读取电影信息
movies = pd.read_csv(
    r"D:\film rec\u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)
# 给用户1推荐电影
# 获取推荐
recommendations = recommend_movies(1, user_movie_matrix, user_similarity_df)

# 转成DataFrame
recommendations = recommendations.reset_index()
recommendations.columns = ["movie_id", "score"]

# 合并电影名字
recommendations = recommendations.merge(movies, on="movie_id")

print("推荐结果：")
print(recommendations[["title", "score"]])

print("test")


