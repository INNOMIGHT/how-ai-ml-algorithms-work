# straight line fitting the model - f(x) = wx + b
import copy
import math

# linear regression with single variable is also called uni-variate linear regression

# x (features) -> f (model) -> y_hat (prediction)

# f_wb = f (function) takes w and b as inputs and depending on w and b, it will predict y_hat

# cost_fn = cost function will tell us how well the model is doing

# w, b = w is weight / slope and b is bias / intercept

# to find the best w and b (closest to target), we construct a cost fn

# general cost fn is finding the error which is y_hat - y. Take average of it. square it. and divide by 2m.
# (calculations purpose)

# cost_fn = 1/2m (sum of all f_wb[i] - y[i])**2

# gradient descent - algorithm to help you find min(J_wb)

# w = w-alpha(derivative of J with respect to w), b = b-alpha(derivative of J with respect to b)

# if you are already on local minimum, w stays same


import matplotlib.pyplot as plt
import numpy as np
import csv

study_data = np.genfromtxt('gpa_study_hours.csv', delimiter=',')

print(study_data)
gpa = study_data[:, 0]
study_time = study_data[:, 1]

print(study_time)
print(gpa)
print(study_time.shape[0])


# cost computation
def compute_cost(study_hours, student_gpas, w, b):
    j_wb = 0
    for i in range(len(study_hours)):
        f_wb = (w * study_hours[i]) + b
        cost_i = (f_wb - student_gpas[i]) ** 2
        j_wb += cost_i

    j_wb = j_wb / (2 * len(study_hours))
    return j_wb


print(compute_cost(study_time, gpa, 2, 2))


# Compute Gradient

def compute_gradient(study_hours, student_gpas, w, b):
    dj_dw = 0
    dj_db = 0

    for i in range(len(study_hours)):
        f_wb = w * study_hours[i] + b
        tmp_dj_dw = (f_wb - student_gpas[i]) * study_hours[i]
        tmp_dj_db = (f_wb - student_gpas[i])

        dj_dw += tmp_dj_dw
        dj_db += tmp_dj_db

    dj_dw = dj_dw / len(study_hours)
    dj_db = dj_db / len(study_hours)

    return dj_dw, dj_db


#
print(compute_gradient(study_time, gpa, 0, 0))


# Apply Gradient Descent

def gradient_descent(study_hours, student_gpas, init_w, init_b, compute_cost_func, compute_gradient_func, alpha,
                     iterations):
    m = len(study_hours)
    j_history = []
    w_history = []
    w = copy.deepcopy(init_w)
    b = init_b

    for i in range(iterations):

        dj_dw, dj_db = compute_gradient_func(study_hours, student_gpas, w, b)

        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)

        if i < 100000:
            cost = compute_cost_func(study_hours, student_gpas, w, b)
            j_history.append(cost)

        if i % math.ceil(iterations / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(j_history[-1]):8.2f}   ")

    return w, b, j_history, w_history


initial_w = 0.2
initial_b = 0.2

learning_rate_alpha = 0.0001
num_iterations = 100000

w, b, j_array, w_array = gradient_descent(study_time, gpa, initial_w, initial_b, compute_cost, compute_gradient,
                                          learning_rate_alpha, num_iterations)
#
print("w = ", w, " and b = ", b, " found by gradient descent")

import joblib

# let us see how well our algorithm works
joblib.dump((w, b), 'gpa_prediction_model.pkl')

m = study_time.shape[0]
predicted_values = np.zeros(m)

for i in range(m):
    predicted_values[i] = w * study_time[i] + b

gpa_check = w * 25 + b
print("for 100 hours of study, a student might get gpa= ", gpa_check)

plt.plot(study_time, predicted_values, c='b')
plt.scatter(study_time, gpa, marker='x', c='r')
plt.title("Student Study Hours VS Their GPA")
plt.ylabel("Student GPA")
plt.xlabel("Student Study Hours")
plt.show()

# def compute_cost(X, y, w, b, m):
#     j_wb = 0
#     for i in range(m):
#         f_wb = np.dot(w[i], X[i+1]) + b
#         cost_i = (f_wb - y[i]) ** 2
#         j_wb += cost_i
#
#     j_wb = j_wb / (2*m)
#     return j_wb
#
#
# print(compute_cost(X, y, initial_w, 0, m))
