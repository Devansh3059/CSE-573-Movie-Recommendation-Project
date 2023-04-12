#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 13:33:54 2023

@author: karanmalik
"""


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import movieposters as mp



df = pd.read_csv('data/u.data', sep='\t', names=['user_id','item_id','rating','timestamp'])
movie_titles = pd.read_csv('data/Movie_Titles.csv',encoding= 'unicode_escape')
df = pd.merge(df, movie_titles, on='item_id')
movies_list = [i.strip().split("::") for i in open('data/ml-1m/movies.dat', 'r',encoding='latin-1').readlines()]
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)


matrix = df.pivot_table(index='user_id', columns='title', values='rating')

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings['number_of_ratings'] = df.groupby('title')['rating'].count()
ratings.sort_values('number_of_ratings', ascending=False).head(20)

data_mean=matrix.mean()
data_std=matrix.std()
normalized_df=(matrix-matrix.mean())/matrix.std()



def getMoviesItemBased(movie_name):
    

    temp=movie_titles[movie_titles['title'].str.contains(movie_name, regex=False)]
    if len(temp)==0:
        return 'Movie not present in database! Please try another one.'
    else:
        full_name=temp.iloc[0,1]
    
        input_movie_rating = normalized_df[full_name]    
        input_movie_similar=normalized_df.corrwith(input_movie_rating)
        
        cosine_movie = pd.DataFrame(input_movie_similar, columns=['Correlation'])
        cosine_movie.dropna(inplace=True)
        cosine_movie = cosine_movie.join(ratings['number_of_ratings'])
            
        final_rec=cosine_movie[cosine_movie['number_of_ratings'] > 20].sort_values(by='Correlation', ascending=False).head(6)
        final_rec.reset_index(inplace=True)
        final_rec=final_rec[['title','number_of_ratings']]
        final_rec.columns=['Title','Number of Ratings']
        
        movie_name,year,poster=[],[],[]

        for x in final_rec['Title'][1:]:
            name=x.split('(')[0][:-1]
            if name[-3:]=='The':
                name=name[-3:]+' ' + name[:-5]
            movie_name.append(name)
            poster.append(mp.get_poster(name))
            
            year.append(x.split('(')[1][:-1])
    return movie_name,year,poster


movie_name,year,poster=getMoviesItemBased('Titanic')










