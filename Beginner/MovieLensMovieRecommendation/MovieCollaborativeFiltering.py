import pandas as pd
from scipy.sparse import csr_matrix

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

#Filter movies with atleast 50 ratings
movies_rc = ratings['movieId'].value_counts()
p_movie = movies_rc[movies_rc >= 50].index
ratings = ratings[ratings['movieId'].isin(p_movie)]

#Filter users rated atleast 20 movies
users_rc = ratings['userId'].value_counts()
exc_users = users_rc[users_rc >=20].index
ratings = ratings[ratings['userId'].isin(exc_users)] 

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
user_item_matrix = user_item_matrix.sub(user_item_matrix.mean(axis=1), axis=0)
#Fill NaN with 0 and convert to sparse matrix
sparse_user_item_matrix = csr_matrix(user_item_matrix.fillna(0).values)

print(sparse_user_item_matrix)