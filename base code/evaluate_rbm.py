#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 00:43:18 2023

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


from sklearn.metrics import mean_squared_error,precision_score,accuracy_score,recall_score
print('RMSE: ',mean_squared_error(new_df['rating'],new_df['prediction'],squared=False))


#Accuracy
ratings_binary=new_df['rating']>2
predictions_binary=new_df['prediction']>2

print('Accuracy: ',accuracy_score(ratings_binary,predictions_binary))

print('Precision: ',precision_score(ratings_binary,predictions_binary))

print('Recall: ',recall_score(ratings_binary,predictions_binary))

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return TP, FP, TN, FN


TP,FP,TN,FN=perf_measure(ratings_binary,predictions_binary)

print('FPR: ',FP/(FP+TN))

print('FNR: ',FN/(TP+FN))
