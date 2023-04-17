#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 00:27:33 2023

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


original = pd.DataFrame(R_demeaned, columns = R_df.columns)

original_ratings=[]
predicted_ratings=[]

for i in range(len(ratings_df)):
    original_ratings.append(original.loc[int(ratings_df.iloc[i,0])-1,str(ratings_df.iloc[i,1])])
    predicted_ratings.append(preds_df.loc[int(ratings_df.iloc[i,0])-1,str(ratings_df.iloc[i,1])])



#EVALUATE

from sklearn.metrics import mean_squared_error,precision_score,accuracy_score,recall_score
print('RMSE: ',mean_squared_error(original_ratings,predicted_ratings,squared=(False)))

or_binary= np.array(original_ratings)>np.mean(original_ratings)
pred_binary= np.array(predicted_ratings)>np.mean(predicted_ratings)

print('Accuracy: ',accuracy_score(or_binary,pred_binary))

print('Precision: ',precision_score(or_binary,pred_binary))

print('Recall; ',recall_score(or_binary,pred_binary))

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


TP,FP,TN,FN=perf_measure(or_binary,pred_binary)

print('FPR: ',FP/(FP+TN))

print('FNR: ',FN/(TP+FN))


#Method 2

# def normalize(values, bounds):
#     return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]

# original_norm=normalize(
#     original_ratings,
#     {'actual':{'lower':min(original_ratings),'upper':max(original_ratings)},'desired':{'lower':1,'upper':5}}
# )

# predicted_norm=normalize(
#     predicted_ratings,
#     {'actual':{'lower':min(predicted_ratings),'upper':max(predicted_ratings)},'desired':{'lower':1,'upper':5}}
# )

# accuracy_score(np.array(original_norm)>3,np.array(predicted_norm)>3)

# precision_score(np.array(original_norm)>3,np.array(predicted_norm)>3)

# recall_score(np.array(original_norm)>3,np.array(predicted_norm)>3)
