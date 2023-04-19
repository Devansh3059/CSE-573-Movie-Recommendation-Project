from movie import app
from flask import Flask, render_template, request, jsonify, url_for, redirect,flash
from movie.forms import MovieForm, UserForm

# copied from item_based

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import movieposters as mp
from imdb import Cinemagoer
ia = Cinemagoer()





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

ratings_list2 = [i.strip().split("::") for i in open('data/ml-1m/ratings.dat', 'r').readlines()]
users_list2 = [i.strip().split("::") for i in open('data/ml-1m/users.dat', 'r').readlines()]
movies_list2 = [i.strip().split("::") for i in open('data/ml-1m/movies.dat', 'r',encoding='latin-1').readlines()]

ratings_df2 = pd.DataFrame(ratings_list2, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'])
movies_df2 = pd.DataFrame(movies_list2, columns = ['MovieID', 'Title', 'Genres'])
movies_df2['MovieID'] = movies_df2['MovieID'].apply(pd.to_numeric)


R_df = ratings_df2.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
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


movie_list,year_list,genre_list,poster_list,rating_list=[],[],[],[],[]

# copied from item_based end


@app.route('/',methods=['GET','POST'])
#@app.route('/home',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/item_based',methods=['GET','POST'])
def itemBasedRec():
    form=MovieForm()
    
    if form.validate_on_submit():
        m=form.moviename.data
        print(m)
        
# copy begin fuction samaan

        temp=movie_titles[movie_titles['title'].str.contains(m, regex=False)]
        
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
            
            
            print('hello')
            movie_list.clear()
            poster_list.clear()
            year_list.clear()
            genre_list.clear()
            rating_list.clear()            

            for x in final_rec['Title'][1:]:
                print('hello2',x)
                name=x.split('(')[0][:-1]
                if name[-3:]=='The':
                    name=name[-3:]+' ' + name[:-5]
                movie_list.append(name)
            
                # poster_list.append(mp.get_poster(name))
                movies = ia.search_movie(name)
                code=movies[0].movieID
                movie = ia.get_movie(code)
                poster_list.append(movie['cover url'])
                genre_list.append(movie['genres'][0])
                rating_list.append(movie['rating'])
                year_list.append(x.split('(')[1][:-1])
            return redirect(url_for('results'))

                
        #print(movie_name,year)

# samaan ends item_based

            #for i in range(1,5):
            #    flash(final[i])
                
    return render_template('item_based.html',form=form)

@app.route('/user_based',methods=['GET','POST'])
def userBasedRec():
    form=UserForm()
    
    if form.validate_on_submit():
        m=form.moviename.data
        m=int(m)
# copy begin fuction samaan

        user_row_number = m - 1 # UserID starts at 1, not 0
        sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
        #print(sorted_user_predictions)
        # Get the user's data and merge in the movie information.
        user_data = ratings_df2[ratings_df2.UserID == (m)]
        user_full = (user_data.merge(movies_df2, how = 'left', left_on = 'MovieID', right_on = 'MovieID').
                        sort_values(['Rating'], ascending=False)
                    )
        
        temp=pd.DataFrame(sorted_user_predictions).reset_index()
        temp['MovieID']=temp['MovieID'].astype(int)
        
        recommendations = (movies_df2[~movies_df2['MovieID'].isin(user_full['MovieID'])].
            merge(temp, how = 'left',
                    left_on = 'MovieID',
                    right_on = 'MovieID').
            rename(columns = {user_row_number: 'Predictions'}).
            sort_values('Predictions', ascending = False).
                            iloc[:5, :-1]
                        )
        movie_name,year,poster=[],[],[]
        movie_list.clear()
        poster_list.clear()
        year_list.clear()
        rating_list.clear()
        for x in recommendations['Title']:
            name=x.split('(')[0][:-1]
            if name[-3:]=='The':
                name=name[-3:]+' ' + name[:-5]
            movie_list.append(name)
            movies = ia.search_movie(name)
            code=movies[0].movieID
            movie = ia.get_movie(code)
            poster_list.append(movie['cover url'])
            
            rating_list.append(movie['rating'])
            #poster_list.append(mp.get_poster(name))
            year_list.append(x.split('(')[1][:-1])
        global genre_list
        genre_list=list(recommendations['Genres'])
        for i in range(len(genre_list)):
            genre_list[i]=genre_list[i].split('|')[0]
        return redirect(url_for('results'))

                
        #print(movie_name,year)

# samaan ends item_based

            #for i in range(1,5):
            #    flash(final[i])
                
    return render_template('user_based.html',form=form)

@app.route('/results',methods=['GET','POST'])
def results():
    #movie_name = request.args.get('movies', None)
    #print(movie_name)
    return render_template('results.html',movie_name=movie_list,posters=poster_list,genres=genre_list,years=year_list,ratings=rating_list)
