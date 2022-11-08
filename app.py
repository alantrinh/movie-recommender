import pandas as pd
from flask import Flask, render_template, request
from recommender import get_nmf_recommendations, get_cf_recommendations

app = Flask(import_name=__name__)

Ratings = pd.read_csv("static/data/ratings_pivot_table.csv",index_col=0)
movies = Ratings.dropna(axis=1, thresh=20).columns

@app.route('/')
def index():
    return render_template('index.html', movies=movies)

@app.route('/recommendations')
def get_recommendations():
    user_query = {key: int(value) for key, value in request.args.to_dict().items() if key != 'movies' and value != ''}
    if len(user_query) < 5:
        model = 'Non-negative Matrix Factorisation'
        recommendations = get_nmf_recommendations(user_query)
    else:
        model = 'Collaborative Filtering'
        recommendations = get_cf_recommendations(user_query)
    return render_template('recommendations.html',recommendations=recommendations, model=model, user_query=user_query)

if __name__ == '__main__':
    app.run(debug=True)
