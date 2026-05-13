import numpy as np
import pandas as pd
import tensorflow as tf


def load_ratings(path):
    data = []

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            user_id, movie_id, rating, _ = line.strip().split("::")
            data.append((int(user_id), int(movie_id), float(rating)))

    return data

def load_movies(path):
    movies = {}

    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")

            movie_id = int(parts[0])
            title = parts[1]

            movies[movie_id - 1] = title  # align with matrix index

    return movies



def create_matrices(data):
    num_users = max(d[0] for d in data)
    num_movies = max(d[1] for d in data)

    Y = np.zeros((num_movies, num_users))
    R = np.zeros((num_movies, num_users))

    for user, movie, rating in data:
        Y[movie - 1, user - 1] = rating
        R[movie - 1, user - 1] = 1

    return Y, R

def initialize(num_movies, num_users, num_features):
    X = np.random.randn(num_movies, num_features)
    # x(i)=feature vector of movie i (Each row of X)
    # X[avengers] = [0.9, 0.1, 0.2]
    # Interpretation
    # High action 
    # Low romance 
    # Some comedy 
    Theta = np.random.randn(num_users, num_features)
    # θ(j)=preference vector of user j (Each row = one user)
    # Theta[user] = [0.8, 0.2, 0.1]
    # Interpretation
    # Likes action 
    # Slightly likes romance
    # Doesn’t care much about comedy
    return X, Theta


def compute_cost(X, Theta, Y, R, lambda_):
    num_movies, num_users = Y.shape
    J = 0.0

    # double loop : Iterating over every possible (movie, user) pair
    for i in range(num_movies):
        for j in range(num_users):
            if R[i, j] == 1: # Only compute where rating exists
                prediction = np.dot(Theta[j], X[i])
                error = prediction - Y[i, j]
                J += 0.5 * (error ** 2) # why squared - penalizes large mistakes more, smooth gradient

    J += (lambda_ / 2) * (np.sum(X**2) + np.sum(Theta**2)) # Regularization - Why BOTH? We control: movie feature magnitude (X),user preference magnitude (Θ)
    # multiplying half d/dx​(1/2​x^2)=x cleaner gradient

    return J

def compute_cost_vectorized(X, Theta, Y, R, lambda_):
    pred = X @ Theta.T
    error = (pred - Y) * R # multiply by R because if there is rating then multiply by 1, if not multiply by zero becomes zero

    J = 0.5 * np.sum(error**2)
    J += (lambda_ / 2) * (np.sum(X**2) + np.sum(Theta**2))

    return J

def compute_rmse(X, Theta, Y, R):
    pred = X @ Theta.T
    error = (pred - Y) * R
    return np.sqrt(np.sum(error**2) / np.sum(R))

# computes “how much to change each parameter” so that prediction error decreases.
def compute_gradients(X, Theta, Y, R, lambda_):
    num_movies, num_users = Y.shape
    num_features = X.shape[1]

    X_grad = np.zeros_like(X) #dJ/dX
    Theta_grad = np.zeros_like(Theta) #dJ/dTheta

    for i in range(num_movies):
        for j in range(num_users):
            if R[i, j] == 1:
                error = np.dot(Theta[j], X[i]) - Y[i, j]

                # Gradient for movie features - Movie moves toward users who liked it
                X_grad[i] += error * Theta[j] # Suppose - (Theta.x - y)^2 = e; derivative dJ/dx = d(1/2e^2)/de . de/dx (chain rule) 
                                                                                    # dJ/dx = e . Theta (remember we assigned error = e)
                # Gradient for user preferences - User moves toward movies they liked
                Theta_grad[j] += error * X[i] # Suppose - (Theta.x - y)^2 = e; derivative dJ/dTheta = d(1/2e^2)/de . de/dTheta (chain rule) 
                                                                                    # dJ/dx = e . x (remember we assigned error = e)

                # This is why it's called collaborative filtering
    # Regularization - Pull values back toward zero
    X_grad += lambda_ * X
    Theta_grad += lambda_ * Theta

    return X_grad, Theta_grad


def compute_gradients_vectorized(X, Theta, Y, R, lambda_):
    pred = X @ Theta.T
    error = (pred - Y) * R

    X_grad = error @ Theta
    Theta_grad = error.T @ X

    X_grad += lambda_ * X
    Theta_grad += lambda_ * Theta

    return X_grad, Theta_grad



def gradient_descent(X, Theta, Y, R, alpha, lambda_, num_iters):
    """
        Optimize movie feature vectors and user preference vectors using batch gradient descent.
    
        This function repeatedly computes gradients for the collaborative filtering
        objective and updates `X` and `Theta` to reduce prediction error on the
        observed ratings indicated by `R`. Progress is logged every 10 iterations
        using the current cost and RMSE.
    
        Args:
            X (np.ndarray): Movie feature matrix of shape (num_movies, num_features).
            Theta (np.ndarray): User preference matrix of shape (num_users, num_features).
            Y (np.ndarray): Ratings matrix of shape (num_movies, num_users).
            R (np.ndarray): Binary indicator matrix of shape (num_movies, num_users),
                where 1 means a rating exists and 0 means missing.
            alpha (float): Learning rate for each update step.
            lambda_ (float): Regularization strength applied to both `X` and `Theta`.
            num_iters (int): Number of gradient descent iterations to run.
    
        Returns:
            tuple[np.ndarray, np.ndarray]: The updated movie feature matrix `X` and
            user preference matrix `Theta`.
        """    
    for iteration in range(num_iters):
        X_grad, Theta_grad = compute_gradients_vectorized(X, Theta, Y, R, lambda_)

        X -= alpha * X_grad
        Theta -= alpha * Theta_grad

        if iteration % 10 == 0:
            cost = compute_cost_vectorized(X, Theta, Y, R, lambda_)
            rmse = compute_rmse(X, Theta, Y, R)
            print(f"Iteration {iteration}, Cost: {cost}, RMSE: {rmse}")
    return X, Theta

if __name__ == "__main__":

    data = load_ratings("data/ratings.dat")
    Y, R = create_matrices(data)
    Y_mean = np.sum(Y, axis=1) / (np.sum(R, axis=1) + 1e-8)
    Y_norm = Y - Y_mean[:, None]
    num_movies, num_users = Y_norm.shape
    X, Theta = initialize(num_movies, num_users, 10)
    J = compute_cost_vectorized(X, Theta, Y_norm, R, lambda_=1.0)
    print("Initial cost without regularization:", J)
    print("Y shape:", Y_norm.shape)
    print("R shape:", R.shape)
    X, Theta = gradient_descent(
    X, Theta, Y_norm, R,
    alpha=1e-5,
    lambda_=1.0,
    num_iters=200
    )
    pred = X @ Theta.T               # X → (movies × features), Theta.T → (features × users)
    pred = pred + Y_mean[:, None]    #pred → (movies × users) eg: pred[i, j] = predicted movie rating for user j for movie i 
    # Also we added mean back. Each movie gets its baseline rating added back

    #Pick user id 0 for prediction
    user_id = 0

    user_predictions = pred[:, user_id] # Get All Predictions for That User. user_predictions[i] = 👉 predicted rating of user 0 for movie i
    rated = R[:, user_id] == 1 # Find Already Rated Movies

    user_predictions[rated] = -np.inf  # Remove Seen Movies they go to the bottom when sorting

    top_indices = np.argsort(user_predictions)[::-1][:10] # Get Top Movies. We use argsort because we care about which items are best, not just what the scores are. Argsort gives indices

    movies = load_movies("data/movies.dat")

    print("\nTop recommendations:\n")
    for idx in top_indices:
        print(movies[idx])