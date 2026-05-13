# hybrid_recommender.py

import numpy as np
import tensorflow as tf

# --- IMPORT CF and CB Recommenders ---
from collaborative_filtering import (
    load_ratings,
    load_movies as load_movies_cf,
    create_matrices,
    initialize,
    compute_cost_vectorized,
    compute_gradients_vectorized,
    gradient_descent
)

from content_based_filtering_neural_networks import (
    load_movies,
    create_movie_features,
    create_user_features,
    create_training_data,
    build_movie_model,
    build_user_model,
    
    build_two_tower_model
)


# Normalization
def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)



# Train Collaborative Filtering
def train_collaborative(data_path):
    data = load_ratings(data_path)

    Y, R = create_matrices(data)

    # Mean normalization
    Y_mean = np.sum(Y, axis=1) / (np.sum(R, axis=1) + 1e-8)
    Y_norm = Y - Y_mean[:, None]

    num_movies, num_users = Y.shape

    X, Theta = initialize(num_movies, num_users, 10)

    print("\nTraining Collaborative Filtering...\n")

    X, Theta = gradient_descent(
        X, Theta, Y_norm, R,
        alpha=1e-5,
        lambda_=1.0,
        num_iters=100
    )

    return X, Theta, Y_mean, R, data



# Step 2: Train Content-Based Model
def train_content_model(ratings_path, movies_path):

    ratings = load_ratings(ratings_path)
    movies, genre_list = load_movies(movies_path)

    valid_movie_ids = set(movies.keys())

    print("Max movie in ratings:", max(r[1] for r in ratings))
    print("Max movie in movies:", max(movies.keys()))

    filtered_ratings = [
        (u, m, r)
        for (u, m, r) in ratings
        if m in valid_movie_ids
    ]
    num_users = max(r[0] for r in ratings) + 1

    movie_features = create_movie_features(movies, genre_list)
    user_features = create_user_features(num_users, filtered_ratings, movie_features)

    user_inputs, movie_inputs, targets = create_training_data(
        filtered_ratings, user_features, movie_features
    )

    print("\nTraining Content-Based Neural Model...\n")

    movie_model = build_movie_model(movie_features.shape[1])
    user_model = build_user_model(user_features.shape[1])

    model = build_two_tower_model(user_model, movie_model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse'
    )

    model.fit(
        [user_inputs, movie_inputs],
        targets,
        epochs=5,
        batch_size=256,
        verbose=1
    )

    return model, user_model, movie_model, user_features, movie_features, filtered_ratings, movies



# Step 3: Hybrid Recommendation
def hybrid_recommend(
    user_id,
    X, Theta, Y_mean, R,
    user_model, movie_model,
    user_features, movie_features,
    movies,
    alpha=0.6
):

    # --- CF scores ---
    pred_cf = X @ Theta.T
    pred_cf = pred_cf + Y_mean[:, None]
    user_cf_scores = pred_cf[:, user_id]

    # --- CB scores ---
    user_vec = user_features[user_id].reshape(1, -1)
    user_embedding = user_model.predict(user_vec, verbose=0)

    movie_embeddings = movie_model.predict(movie_features, verbose=0)
    user_cb_scores = (user_embedding @ movie_embeddings.T).flatten()

    # --- Normalize ---
    user_cf_scores = normalize(user_cf_scores)
    user_cb_scores = normalize(user_cb_scores)

    # --- Combine ---
    hybrid_scores = alpha * user_cf_scores + (1 - alpha) * user_cb_scores

    # --- Remove seen ---
    rated = R[:, user_id] == 1
    hybrid_scores[rated] = -np.inf

    # --- Top ---
    top_indices = np.argsort(hybrid_scores)[::-1][:10]

    print("\nHybrid Recommendations:\n")
    for idx in top_indices:
        print(movies[idx]["title"])



# MAIN
if __name__ == "__main__":

    ratings_path = "data/ratings.dat"
    movies_path = "data/movies.dat"

    # --- Train CF ---
    X, Theta, Y_mean, R, data = train_collaborative(ratings_path)

    # --- Train Content Model ---
    model, user_model, movie_model, user_features, movie_features, ratings, movies = train_content_model(
        ratings_path, movies_path
    )

    # --- Recommend ---
    hybrid_recommend(
        user_id=0,
        X=X,
        Theta=Theta,
        Y_mean=Y_mean,
        R=R,
        user_model=user_model,
        movie_model=movie_model,
        user_features=user_features,
        movie_features=movie_features,
        movies=movies,
        alpha=0.6
    )