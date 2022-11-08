import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

Ratings = pd.read_csv("static/data/ratings_pivot_table.csv",index_col=0)

with open('static/model/nmf_model_movielens.pkl','rb') as file:
    nmf_model = pickle.load(file)

def get_new_user_df(query:dict, columns):
    return pd.DataFrame(
        data=query,
        columns=columns,
        index = ['new_user']
    )

def calculate_nmf_model_scores(df_new_user):
    df_new_user_imputed = df_new_user.fillna(Ratings.mean())

    Q_matrix = nmf_model.components_

    P_new_user_matrix = nmf_model.transform(df_new_user_imputed)
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q_matrix)
    return pd.DataFrame(
        data=R_hat_new_user_matrix,
        columns=nmf_model.feature_names_in_,
        index = ['new_user']
    )

def get_nmf_recommendations(query:dict):
    """returns movie recommendations based on NMF algorithm"""
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """

    # 1. candidate generation
    
    # construct new_user-item dataframe given the query
    df_new_user = get_new_user_df(query, nmf_model.feature_names_in_)

    # 2. scoring
    R_hat_new_user_df = calculate_nmf_model_scores(df_new_user)
    
    # 3. ranking 
    recommendations = list(R_hat_new_user_df.transpose()
        .drop(query.keys())  # filter out movies already seen by the user
        .sort_values(ascending=False, by='new_user') # return the top-10 highest rated movie ids or titles
        .head(10).index
    )
    
    return recommendations

def get_user_similarities(Ratings_with_new_user):
    user_similarity = cosine_similarity(Ratings_with_new_user.T)
    return pd.DataFrame(
        user_similarity,
        columns = Ratings_with_new_user.columns,
        index = Ratings_with_new_user.columns
    ).round(2)

def calculate_cf_model_scores(Ratings_with_new_user, movies, df_new_user):
    # get user similarity matrix DataFrame
    user_similarity_df = get_user_similarities(Ratings_with_new_user)
    # candidates are all unseen movies
    candidates = movies[df_new_user.T['new_user'].isna()]
    # take top 50 neighbours
    neighbours = user_similarity_df['new_user'].sort_values(ascending=False).index[1:51]

    unseen_movie_predicted_ratings = {}

    for unseen_movie in candidates:
        # extract users who have rated unseen movies
        current_movie_reviewers = Ratings.T.columns[~Ratings.T.loc[unseen_movie].isna()]
        # we want to create an intersection, so we save the users as a set-object
        current_movie_reviewers = set(current_movie_reviewers)

        intersection = set(neighbours).intersection(current_movie_reviewers)
        # only consider if at least 10 neighbours have reviewed the unseen movie
        if len(intersection) >= 5:
            num = 0
            den = 0
            for neighbour in intersection:
                ratings = Ratings_with_new_user[neighbour][unseen_movie] 
                sim_score = user_similarity_df['new_user'][neighbour]
                num = num + (ratings*sim_score)
                den = den + sim_score
            
            pred_ratings = num/den
            unseen_movie_predicted_ratings[unseen_movie] = pred_ratings
        else:
            unseen_movie_predicted_ratings[unseen_movie] = 0

    return unseen_movie_predicted_ratings

def get_cf_recommendations(query:dict):
    # recommend from movies with at least 20 ratings
    movies = Ratings.dropna(axis=1, thresh=20).columns 

    df_new_user = get_new_user_df(query, movies)
    Ratings_with_new_user = Ratings.T.join(df_new_user.T).fillna(0)

    unseen_movie_ratings = calculate_cf_model_scores(Ratings_with_new_user, movies, df_new_user)

    #return top 10 movies
    return [movie for movie, _ in sorted(unseen_movie_ratings.items(), key=lambda item: item[1], reverse=True)][:10]

if __name__ == '__main__':
    print(get_nmf_recommendations())
