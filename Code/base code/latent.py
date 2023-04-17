#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 16:20:17 2023

@author: karanmalik
"""

import pandas as pd
import numpy as np
import movieposters as mp


ratings_list = [i.strip().split("::") for i in open('data/ml-1m/ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('data/ml-1m/users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('data/ml-1m/movies.dat', 'r',encoding='latin-1').readlines()]

ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)


R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
R_df.head()

R = R_df.to_numpy()
R=R.astype(int)
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)


from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)

sigma = np.diag(sigma)


all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)


def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    #print(sorted_user_predictions)
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                     sort_values(['Rating'], ascending=False)
                 )
    
    temp=pd.DataFrame(sorted_user_predictions).reset_index()
    temp['MovieID']=temp['MovieID'].astype(int)
    
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
          merge(temp, how = 'left',
                left_on = 'MovieID',
                right_on = 'MovieID').
          rename(columns = {user_row_number: 'Predictions'}).
          sort_values('Predictions', ascending = False).
                        iloc[:num_recommendations, :-1]
                      )
    movie_name,year,poster=[],[],[]

    for x in recommendations['Title']:
        name=x.split('(')[0][:-1]
        if name[-3:]=='The':
            name=name[-3:]+' ' + name[:-5]
        movie_name.append(name)
        poster.append(mp.get_poster(name))
        year.append(x.split('(')[1][:-1])
    genre=list(recommendations['Genres'])
    
    return movie_name,year,genre,poster

movie_name,year,genre,posters= recommend_movies(preds_df, 99, movies_df, ratings_df, 5)

