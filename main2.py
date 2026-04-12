import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 读取数据 + 划分训练/测试
ratings = pd.read_csv(
    "u.data",
    sep="\t",
    names=["user_id", "movie_id", "rating", "timestamp"]
)

train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)


# 用训练集构建用户-电影矩阵
user_movie_matrix = train_df.pivot_table(
    index="user_id",
    columns="movie_id",
    values="rating"
)
user_movie_matrix = user_movie_matrix.fillna(0)

# 计算用户相似度
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)


# 冷启动：热门榜兜底（新用户用）
global_mean = float(train_df["rating"].mean())

movie_stats = train_df.groupby("movie_id")["rating"].agg(["mean", "count"]).reset_index()   # 这里reset_index把 movie_id 从 index 变成普通列，方便后面 merge
# 门槛避免“少量用户打分导致的虚高”
min_count = 20 #至少有20个人评分
popular_movies = movie_stats[movie_stats["count"] >= min_count].sort_values(["mean", "count"], ascending=[False,False]) #第一排序mean，第二排序count，都为降序 这里用到了布尔筛选，和note里的布尔索引是一样的用法，判断式得到的是True/false，然后保留true

# 用 movie_mean 作为“热门榜分数”
popular_scores_series = popular_movies.set_index("movie_id")["mean"].sort_values(ascending=False) 

if len(popular_scores_series) == 0:
    # 万一阈值太严格，比如min_count设置的值过高
    popular_scores_series = pd.Series(dtype=float) #创建一个空的浮点类型的series，即空的推荐结果，防止后续操作报错


# 核心推荐函数
def recommend_movies(user_id, user_movie_matrix, user_similarity_df, top_n=5, top_users_n=10):
    # 冷启动：新用户（训练集中没有）
    if user_id not in user_movie_matrix.index:
        if len(popular_scores_series) >= top_n:
            return popular_scores_series.head(top_n)
        # 如果热门榜不够，再用全局均值补齐（用虚拟 score）
        out = popular_scores_series.copy() #创建副本是为了不破坏原本热门榜数据，因为如果out直接等于popular_scores_series的话，后面修改out会影响原数据
        need = top_n-len(out)
        if need > 0:
            out = pd.concat([out, pd.Series([global_mean] * need, index=[])]).sort_values(ascending=False) #补的index是空的（会显示NaN?)，评分是全局均值，没有真实的movieid，只是为了保证推荐数量
        return out.head(top_n)

    # 1. 找到和该用户相似的用户
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    # 去掉自己
    similar_users = similar_users.drop(user_id)
    # 2. 取前若干相似用户
    top_users = similar_users.head(top_users_n).index

    # 3. 这些用户喜欢的电影（评分高的）——沿用原来 mean(axis=0)
    similar_users_movies = user_movie_matrix.loc[top_users]
    movie_scores = similar_users_movies.mean(axis=0)

    # 4. 去掉该用户已经看过的电影
    user_movies = user_movie_matrix.loc[user_id]
    unseen_movie_ids = user_movies[user_movies == 0].index

    # 5. 只保留没看过的电影
    recommendations = movie_scores.loc[unseen_movie_ids]

    # 6. 可选：过滤掉完全没有邻居评分的 0 分（让推荐更合理）
    recommendations = recommendations[recommendations > 0]

    # 7. 排序取 top_n
    recommendations = recommendations.sort_values(ascending=False).head(top_n)

    # 8. 如果不够 top_n，用热门榜补齐（保证 API/评估时长度稳定）
    if len(recommendations) < top_n and len(popular_scores_series) > 0:
        missing = top_n - len(recommendations)
        already = set(recommendations.index.tolist()) #已经推荐的电影的集合
        popular_to_add = popular_scores_series[~popular_scores_series.index.isin(already)].head(missing) #去重后取前misssing个
        recommendations = pd.concat([recommendations, popular_to_add], axis=0).sort_values(ascending=False).head(top_n) #最后再排一次序

    return recommendations



# 用于评估：预测单个 (user, movie) 的评分
def predict_rating(user_id, movie_id, user_movie_matrix, user_similarity_df, top_users_n=10):
    # 用户冷启动
    if user_id not in user_movie_matrix.index:
        return global_mean

    # 电影冷启动（训练集中没出现过该 movie）
    if movie_id not in user_movie_matrix.columns:
        return global_mean

    # 找相似用户
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).drop(user_id)
    top_users = similar_users.head(top_users_n).index

    # 邻居对该电影的评分
    neighbor_ratings = user_movie_matrix.loc[top_users, movie_id]

    # 只用有效评分（>0）
    valid = neighbor_ratings[neighbor_ratings > 0]
    if len(valid) == 0:
        # 没邻居评分：回退电影均值（如果有）
        movie_nonzero = user_movie_matrix[movie_id]
        movie_nonzero = movie_nonzero[movie_nonzero > 0]
        if len(movie_nonzero) > 0:
            return float(movie_nonzero.mean())
        return global_mean

    # 预测：沿用 recommend_movies 的思路，返回邻居均值
    return float(valid.mean())



# 评估指标：MAE / RMSE / Precision@K / Recall@K
def evaluate_predictions_mae_rmse():
    y_true = [] #真实评分，注意是空的list
    y_pred = [] #预测评分

    # 注意：我们在 test_df 上逐条预测
    for row in test_df.itertuples(index=False):
        pred = predict_rating(
            row.user_id,
            row.movie_id,
            user_movie_matrix,
            user_similarity_df
        ) #读取当前一轮的userid和movieid然后预测
        y_true.append(row.rating)
        y_pred.append(pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return mae, rmse


def evaluate_precision_recall_at_k(K=10, rel_threshold=4.0):
    # 对每个用户：看 test 里哪些是“正样本”（评分>=阈值表示喜爱这部电影）
    precisions = []
    recalls = []

    for user_id, user_test in test_df.groupby("user_id"): #注意是测试集
        relevant = set(user_test[user_test["rating"] >= rel_threshold]["movie_id"].tolist()) #喜爱的电影
        if len(relevant) == 0: #用户没有喜欢的电影
            continue #跳过，下一个用户

        rec_series = recommend_movies(
            user_id,
            user_movie_matrix,
            user_similarity_df,
            top_n=K
        )
        recommended = set(rec_series.index.tolist())

        hits = len(recommended & relevant) #推荐中命中的数量
        precisions.append(hits / K) #推荐里多少是对的
        recalls.append(hits / len(relevant)) #用户喜欢的你找到了多少

    if len(precisions) == 0:
        return 0.0, 0.0

    return float(np.mean(precisions)), float(np.mean(recalls))


mae, rmse = evaluate_predictions_mae_rmse() 
precision_at_k, recall_at_k = evaluate_precision_recall_at_k(K=10, rel_threshold=4.0)

print(f"MAE: {mae:.4f}") #.4f表示保留是
print(f"RMSE: {rmse:.4f}")
print(f"Precision@10: {precision_at_k:.4f}")
print(f"Recall@10: {recall_at_k:.4f}")


# 读取电影信息，并输出一个用户的推荐（沿用原来的输出方式）
movies = pd.read_csv(
    r"D:\film rec\u.item",
    sep="|",
    encoding="latin-1",
    header=None,
    usecols=[0, 1],
    names=["movie_id", "title"]
)

recommendations = recommend_movies(1, user_movie_matrix, user_similarity_df)
recommendations = recommendations.reset_index()
recommendations.columns = ["movie_id", "score"]

recommendations = recommendations.merge(movies, on="movie_id")

print("推荐结果：")
print(recommendations[["title", "score"]])
