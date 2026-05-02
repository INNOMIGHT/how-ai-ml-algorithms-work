import numpy as np

# extract genres
def extract_genres(movies_path):
    genres_set = set()

    with open(movies_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")
            genres = parts[2].split("|")
            genres_set.update(genres)

    return sorted(list(genres_set))


def load_movies(path):
    movies = {}

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")

            movie_id = int(parts[0])
            title = parts[1]

            movies[movie_id - 1] = title  # align with matrix index

    return movies


def load_ratings(path):
    data = []

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            user_id, movie_id, rating, _ = line.strip().split("::")
            data.append((int(user_id), int(movie_id), float(rating)))

    return data


def create_matrices(data):
    num_users = max(d[0] for d in data)
    num_movies = max(d[1] for d in data)

    Y = np.zeros((num_movies, num_users))
    R = np.zeros((num_movies, num_users))

    for user, movie, rating in data:
        Y[movie - 1, user - 1] = rating
        R[movie - 1, user - 1] = 1

    return Y, R


# convert each movie to a binary vector 
# Movie Feature Matrix:
# Toy Story → [0, 0, 1, 1, 0, ...]
# Heat      → [1, 0, 0, 0, 1, ...]
def create_movie_features(movies_path, genre_list):
    genre_index = {genre: i for i, genre in enumerate(genre_list)}

    num_movies = 3952  # from dataset
    num_features = len(genre_list)

    X = np.zeros((num_movies, num_features))

    with open(movies_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")

            movie_id = int(parts[0]) - 1
            genres = parts[2].split("|")

            for genre in genres:
                idx = genre_index[genre]
                X[movie_id, idx] = 1

    return X


# User profile = weighted average of liked movies
def build_user_profile(user_id, X, Y, R):
    rated = R[:, user_id] == 1
    ratings = Y[rated, user_id]
    movies = X[rated]

    # weighted sum
    profile = np.dot(ratings, movies) / (np.sum(ratings) + 1e-8)

    return profile


# predict scores
def predict_scores(user_profile, X):
    return X @ user_profile



 
# recommend
if __name__ == "__main__":

    # 🔹 Load ratings data
    data = load_ratings("data/ratings.dat")
    Y, R = create_matrices(data)

    print("Y shape:", Y.shape)
    print("R shape:", R.shape)

    # 🔹 Extract genres
    genre_list = extract_genres("data/movies.dat")
    print("Number of genres:", len(genre_list))

    # 🔹 Create movie feature matrix
    X = create_movie_features("data/movies.dat", genre_list)
    print("Movie feature matrix shape:", X.shape)

    # 🔹 Pick a user
    user_id = 0

    # 🔹 Build user profile
    user_profile = build_user_profile(user_id, X, Y, R)

    print("\nUser profile vector:\n", user_profile)

    # 🔹 Predict scores
    scores = predict_scores(user_profile, X)

    # 🔹 Remove already rated movies
    rated = R[:, user_id] == 1
    scores[rated] = -np.inf

    # 🔹 Get top recommendations
    top_indices = np.argsort(scores)[::-1][:10]

    # 🔹 Load movie names
    movies = load_movies("data/movies.dat")

    print("\nTop recommendations:\n")
    for idx in top_indices:
        print(movies[idx])