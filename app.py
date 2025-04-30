from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# -------------------- MOCK DATA --------------------

movies = pd.DataFrame({
    'movie_id': range(1, 11),
    'title': [f"Film {i}" for i in range(1, 11)],
    'genres': [
        "Action Adventure", "Romance Comedy", "Drama", "Action Sci-Fi", "Comedy",
        "Documentary", "Horror Thriller", "Fantasy Adventure", "Romance Drama", "Action Thriller"
    ]
})

ratings_data = [
    (1, 1, 5), (1, 2, 4), (1, 3, 4),
    (2, 2, 5), (2, 3, 3), (2, 4, 4),
    (3, 1, 4), (3, 4, 5), (3, 5, 2),
    (4, 1, 2), (4, 2, 2), (4, 5, 5),
    (5, 3, 5), (5, 4, 4), (5, 5, 4)
]
ratings = pd.DataFrame(ratings_data, columns=["user_id", "movie_id", "rating"])

# -------------------- RECOMMENDER --------------------

def get_recommendations(user_id):
    user_movie_matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)

    # KNN Collaborative
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(user_movie_matrix)
    query_index = user_id - 1
    distances, indices = model_knn.kneighbors(user_movie_matrix.iloc[query_index, :].values.reshape(1, -1), n_neighbors=3)
    neighbor_ids = user_movie_matrix.index[indices.flatten()[1:]]

    neighbor_ratings = ratings[ratings['user_id'].isin(neighbor_ids)]
    movie_scores = neighbor_ratings.groupby('movie_id')['rating'].mean().sort_values(ascending=False)
    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id']
    collab_recs = movie_scores[~movie_scores.index.isin(rated_movies)]
    collab_df = collab_recs.reset_index().rename(columns={"rating": "collab_score"})

    # Content-based
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    user_ratings = ratings[(ratings['user_id'] == user_id) & (ratings['rating'] >= 4)]
    liked_ids = user_ratings['movie_id'].tolist()
    content_scores = np.mean(cosine_sim[liked_ids], axis=0)

    content_df = pd.DataFrame({
        'movie_id': movies['movie_id'],
        'content_score': content_scores
    })
    content_df = content_df[~content_df['movie_id'].isin(liked_ids)]

    # Normalize and combine
    scaler = MinMaxScaler()
    collab_df['collab_score'] = scaler.fit_transform(collab_df[['collab_score']])
    content_df['content_score'] = scaler.fit_transform(content_df[['content_score']])

    hybrid_df = pd.merge(collab_df, content_df, on='movie_id', how='outer').fillna(0)
    hybrid_df['hybrid_score'] = 0.5 * hybrid_df['collab_score'] + 0.5 * hybrid_df['content_score']
    final = pd.merge(hybrid_df, movies, on='movie_id')
    final = final.sort_values(by='hybrid_score', ascending=False).head(5)
    return final[['title', 'genres', 'hybrid_score']]

# -------------------- ROUTES --------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        recommendations = get_recommendations(user_id).to_dict(orient='records')
    return render_template('index.html', recommendations=recommendations)

# -------------------- RUN --------------------

if __name__ == '__main__':
    app.run(debug=True)