#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:41:14 2023

@author: karanmalik
"""

from recommenders.models.rbm.rbm import RBM
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.datasets import movielens
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.timer import Timer
from recommenders.utils.plot import line_graph

import sys

import pandas as pd
import numpy as np
import tensorflow as tf


MOVIELENS_DATA_SIZE = '1m'
data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    header=['userID','movieID','rating','timestamp']
)

data.head()

header = {
        "col_user": "userID",
        "col_item": "movieID",
        "col_rating": "rating",
    }

#instantiate the sparse matrix generation  
am = AffinityMatrix(df = data, **header)


#obtain the sparse matrix 
X, _, z = am.gen_affinity_matrix()

#Xtr, Xtst = numpy_stratified_split(X)


model = RBM(
    possible_ratings=np.setdiff1d(np.unique(X), np.array([0])),
    visible_units=X.shape[1],
    hidden_units=100,
    training_epoch=10,
    minibatch_size=100,
    keep_prob=0.9,
    with_metrics=True
)

model.save(file_path='rbm_model.ckpt')

with Timer() as train_time:
    model.fit(X)


def getMovieRatingRBM(user_id):
    
    
    with Timer() as prediction_time:
        top_k =  model.recommend_k_items(X)
        
    
    top_k_df = am.map_back_sparse(top_k, kind = 'prediction')
    test_df = am.map_back_sparse(Xtst, kind = 'ratings')
    
    top_k_df=top_k_df[top_k_df['userID']==user_id]
    top_k_df=top_k_df.sort_values(by=['prediction'],ascending=False)
    top_k_df.drop('userID',axis=1,inplace=True)
    return top_k_df[:5]

getMovieRatingRBM(1)

##########


#EVALUATIONS

X, _, z = am.gen_affinity_matrix()



model = RBM(
    possible_ratings=np.setdiff1d(np.unique(X), np.array([0])),
    visible_units=X.shape[1],
    hidden_units=600,
    training_epoch=100,
    minibatch_size=8,
    keep_prob=0.9,
    with_metrics=True
)

with Timer() as train_time:
    model.fit(X)

pred = am.map_back_sparse(model.predict(X), kind = 'prediction')

new_df = pd.merge(data, pred,  how='inner', left_on=['userID','movieID'], right_on = ['userID','movieID'])

    
#RMSE
from sklearn.metrics import mean_squared_error,precision_score,accuracy_score,recall_score
mean_squared_error(new_df['rating'],new_df['prediction'],squared=False)


#Accuracy
ratings_binary=new_df['rating']>2
predictions_binary=new_df['prediction']>2
accuracy_score(ratings_binary,predictions_binary)


precision_score(ratings_binary,predictions_binary)

recall_score(ratings_binary,predictions_binary)


