import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
'''
#Filter movies with atleast 50 ratings
movies_rc = ratings['movieId'].value_counts()
p_movie = movies_rc[movies_rc >= 50].index
ratings = ratings[ratings['movieId'].isin(p_movie)]

#Filter users rated atleast 20 movies
users_rc = ratings['userId'].value_counts()
exc_users = users_rc[users_rc >=20].index
ratings = ratings[ratings['userId'].isin(exc_users)] 
'''
ratings = ratings[ratings['userId']<= 10000]
'''
print(movies.head())
print(ratings.head())
print(movies.info())
print(ratings.info())
'''
#merge dataframes
data = pd.merge(ratings, movies, on='movieId')

#create a user-item matrix
user_item_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')
users_with_no_ratings = user_item_matrix.isna().all(axis=1)
users_with_few_ratings = user_item_matrix.count(axis=1) < 20  # Threshold: fewer than 20 ratings
user_item_matrix_normalized = user_item_matrix.copy()
for user in user_item_matrix.index:
    if not users_with_no_ratings[user] and not users_with_few_ratings[user]:
        user_mean = user_item_matrix.loc[user].mean()
        user_item_matrix_normalized.loc[user] = user_item_matrix.loc[user] - user_mean
#user_item_matrix = user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0)
#Fill NaN with 0 and convert to sparse matrix
sparse_user_item_matrix = csr_matrix(user_item_matrix_normalized.fillna(0).values)

#print(sparse_user_item_matrix)

#Find user similarity
user_similarity = cosine_similarity(sparse_user_item_matrix)
dfuser_similarity = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

#print(dfuser_similarity.head())

def recommend_movies(user_id, n_recommend=5) :
    # Check if the user has no ratings or very few ratings
    if users_with_no_ratings[user_id] or users_with_few_ratings[user_id]:
        print("User has very few or no ratings. Recommending popular movies.")
        popular_movies = ratings['movieId'].value_counts()
        popular_movies = popular_movies[popular_movies >= 1000].sort_values(ascending=False).head(n_recommend).index
        #popular_movies = popular_movies[popular_movies >= 1000].head(n_recommend).index
        return movies[movies['movieId'].isin(popular_movies)]['title']
    #Get the similarity scores for the target user
    similar_users = dfuser_similarity[user_id].sort_values(ascending = False)
    # Debug: Print the top similar users
    print(f"Top similar users for user {user_id}:")
    print(similar_users.head())
    #Get movies rated by similar users but not the target user
    target_user_rating = user_item_matrix_normalized.loc[user_id]
    unseen_movies = target_user_rating[target_user_rating.isna()].index
    # Debug: Print the number of unseen movies
    print(f"Number of unseen movies for user {user_id}: {len(unseen_movies)}")
    print("Target user's rating (samples):")
    print(target_user_rating.head())
    print(f"NaN values in target user's ratings: {target_user_rating.isna().sum()}")
    print(f"Zero values in target user's ratings: {target_user_rating.eq(0).sum()}")
    #Check for unseen movies by the target
    if len(unseen_movies) == 0:
        print("User has rated all movies. Recommending popular movies.")
        popular_movies = ratings['movieId'].value_counts()
        popular_movies = popular_movies[popular_movies >= 1000].sort_values(ascending=False).head(n_recommend).index
        return movies[movies['movieId'].isin(popular_movies)]['title']

    #Aggregate ratings from similar users for unseen movies
    recommendations = user_item_matrix_normalized.loc[similar_users.index, unseen_movies].mean(axis=0)
    #Filter out niche movies
    movie_rating_counts = ratings['movieId'].value_counts()
    recommendations = recommendations[movie_rating_counts[recommendations.index] >= 500]
    #Sort by highest rated movies
    recommendations = recommendations.sort_values(ascending = False).head(n_recommend).index
    print("Recommend movies based on other user of same preferences : ")
    return movies[movies['movieId'].isin(recommendations)]['title']

print(recommend_movies(5193))
