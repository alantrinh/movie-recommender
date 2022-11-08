import pandas as pd
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity

Ratings = pd.read_csv("static/data/ratings_pivot_table.csv",index_col=0)

with open('static/model/nmf_model_movielens.pkl','rb') as file:
    nmf_model = pickle.load(file)

def get_nmf_recommendations(query:dict):
    """returns movie recommendations based on NMF algorithm"""
    """
    Filters and recommends the top k movies for any given input query based on a trained NMF model. 
    Returns a list of k movie ids.
    """

    print('**query**', query)
    # 1. candidate generation
    
    # construct new_user-item dataframe given the query
    df_new_user = pd.DataFrame(
        data=query,
        columns=nmf_model.feature_names_in_,
        index = ['new_user']
    )

    df_new_user_imputed = df_new_user.fillna(Ratings.mean())

    # 2. scoring
    
    # calculate the score with the NMF model
    Q_matrix = nmf_model.components_

    P_new_user_matrix = nmf_model.transform(df_new_user_imputed)
    R_hat_new_user_matrix = np.dot(P_new_user_matrix, Q_matrix)
    R_hat_new_user_df = pd.DataFrame(
        data=R_hat_new_user_matrix,
        columns=nmf_model.feature_names_in_,
        index = ['new_user']
    )
    
    # 3. ranking
    
    recommendations = list(R_hat_new_user_df.transpose()
        .drop(query.keys())  # filter out movies already seen by the user
        .sort_values(ascending=False, by='new_user') # return the top-10 highest rated movie ids or titles
        .head(10).index
    )
    
    return recommendations

def get_cf_recommendations(query:dict):
    # recommend from movies with at least 20 ratings
    movies = Ratings.dropna(axis=1, thresh=20).columns 
    df_new_user = pd.DataFrame(
        data=query,
        columns=movies,
        index = ['new_user']
    )
    Ratings_with_new_user = Ratings.T.join(df_new_user.T).fillna(0)

    user_similarity = cosine_similarity(Ratings_with_new_user.T)
    user_similarity = pd.DataFrame(user_similarity, columns = Ratings_with_new_user.columns, index = Ratings_with_new_user.columns).round(2)

    candidates = movies[df_new_user.T['new_user'].isna()]
    # take top 50 neighbours
    neighbours = user_similarity['new_user'].sort_values(ascending=False).index[1:51]

    predictions = {}

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
                sim_score = user_similarity['new_user'][neighbour]
                num = num + (ratings*sim_score)
                den = den + sim_score
            
            pred_ratings = num/den
            predictions[unseen_movie] = pred_ratings
        else:
            predictions[unseen_movie] = 0

    #return top 10 movies
    return [movie for movie, _ in sorted(predictions.items(), key=lambda item: item[1], reverse=True)][:10]

if __name__ == '__main__':
    print(get_nmf_recommendations())
