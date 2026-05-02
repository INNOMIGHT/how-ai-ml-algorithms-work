import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


def load_ratings(path):
    data = []

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            user_id, movie_id, rating, _ = line.strip().split("::")
            data.append((int(user_id)-1, int(movie_id)-1, float(rating)))

    return data


def load_movies(path):
    movies = {}
    genre_set = set()

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")
            movie_id = int(parts[0]) - 1
            title = parts[1]
            genres = parts[2].split("|")

            movies[movie_id] = {"title": title, "genres": genres}
            genre_set.update(genres)

    genre_list = sorted(list(genre_set))
    return movies, genre_list


# Toy Story → [Animation=1, Comedy=1, others=0]
def create_movie_features(movies, genre_list):
    genre_index = {g: i for i, g in enumerate(genre_list)}

    num_movies = max(movies.keys()) + 1
    num_features = len(genre_list)

    X = np.zeros((num_movies, num_features))

    for movie_id, info in movies.items():
        for g in info["genres"]:
            X[movie_id, genre_index[g]] = 1

    return X


# [Action=high, Romance=low, Comedy=medium]
def create_user_features(num_users, ratings, movie_features):
    num_features = movie_features.shape[1]
    user_features = np.zeros((num_users, num_features))

    for user, movie, rating in ratings:
        user_features[user] += rating * movie_features[movie]

    # normalize
    norms = np.linalg.norm(user_features, axis=1, keepdims=True) + 1e-8
    user_features = user_features / norms

    return user_features


# (user_features, movie_features) → rating Eg: [0.2, 0.8, ...], [1,0,0,...] -> 5
def create_training_data(ratings, user_features, movie_features):
    user_inputs = []
    movie_inputs = []
    targets = []

    for user, movie, rating in ratings:
        user_inputs.append(user_features[user])
        movie_inputs.append(movie_features[movie])
        targets.append(rating)

    return (
        np.array(user_inputs),
        np.array(movie_inputs),
        np.array(targets)
    )


# movie tower (compare with blog)
def build_movie_model(num_features, embedding_dim=32):
    inputs = tf.keras.Input(shape=(num_features,))

    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(embedding_dim)(x)

    return Model(inputs, outputs, name="movie_model")


# user tower 
def build_user_model(num_features, embedding_dim=32):
    inputs = tf.keras.Input(shape=(num_features,))

    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(32, activation='relu')(x)

    outputs = layers.Dense(embedding_dim)(x)

    return Model(inputs, outputs, name="user_model")


# two tower model as seen in blog 
def build_two_tower_model(user_model, movie_model):

    user_input = tf.keras.Input(shape=user_model.input_shape[1:])
    movie_input = tf.keras.Input(shape=movie_model.input_shape[1:])

    user_embedding = user_model(user_input)
    movie_embedding = movie_model(movie_input)

    # Dot product computes score=Vu​⋅Vm (user, movie) → predicted rating
    dot_product = layers.Dot(axes=1)([user_embedding, movie_embedding])

    return tf.keras.Model([user_input, movie_input], dot_product)


if __name__ == "__main__":

    # Load data
    ratings = load_ratings("data/ratings.dat")
    movies, genre_list = load_movies("data/movies.dat")

    num_users = max(r[0] for r in ratings) + 1

    # Features
    movie_features = create_movie_features(movies, genre_list)
    user_features = create_user_features(num_users, ratings, movie_features)

    # Training data
    user_inputs, movie_inputs, targets = create_training_data(
        ratings, user_features, movie_features
    )

    print("User input shape:", user_inputs.shape)
    print("Movie input shape:", movie_inputs.shape)

    # Build models
    movie_model = build_movie_model(movie_features.shape[1])
    user_model = build_user_model(user_features.shape[1])

    model = build_two_tower_model(user_model, movie_model)

    # optimizer -> how weights update
    #loss -> how error is measured
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse'
    )

    # Train For each batch:
    # forward pass
    # compute prediction
    # compute loss
    # backpropagation
    # update weights
    # automatic gradient descent
    model.fit(
        [user_inputs, movie_inputs],
        targets,
        epochs=5,
        batch_size=256
    )

    # Vectorised Inference (recommend for user 0)

    user_id = 0

    # User embedding
    user_vec = user_features[user_id].reshape(1, -1)
    user_embedding = user_model.predict(user_vec, verbose=0)

    # All movie embeddings
    movie_embeddings = movie_model.predict(movie_features, verbose=0)

    # Compute scores
    scores = user_embedding @ movie_embeddings.T
    scores = scores.flatten()

    # Remove seen movies
    seen = [m for u, m, _ in ratings if u == user_id]
    scores[seen] = -np.inf

    # Top recommendations
    top_indices = np.argsort(scores)[::-1][:10]

    print("\nTop recommendations:\n")
    for idx in top_indices:
        print(movies[idx]["title"])
