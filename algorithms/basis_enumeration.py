import numpy as np
import itertools
from sklearn.preprocessing import normalize
import time

EPSILON = 1e-10


def basis_enumeration(S, timeout=None, known_sol=None, tol=1e-6):
    S = normalize(S, axis=0)  # Normalize the input
    n, d = S.shape
    ones = np.ones(n)
    cur_best_cm = 10

    # Start time for timeout
    start_time = time.time() if timeout is not None else None

    # Iterate through all n-dimensional subsets of S
    for i, sub in enumerate(itertools.combinations(S.T, n)):

        # Check for timeout if specified
        if timeout is not None and time.time() - start_time >= timeout:
            break
        if known_sol and abs(cur_best_cm - known_sol) < tol:
            break

        sub = np.array(sub)
        determinant = np.linalg.det(sub)

        # Check if it's a valid basis (non-singular)
        if abs(determinant) > EPSILON:
            sub = sub.T

            # Compute Gram matrix
            gram = sub.T @ sub
            gamma = 1 / np.sqrt(ones.T @ np.linalg.inv(gram) @ ones)

            # Compute candidate cosine vector
            unit_vec = np.linalg.inv(sub.T) @ (gamma * ones)
            dot_vec = unit_vec.T @ S

            # Compute the candidate cosine measure
            max_val = np.max(dot_vec)
            cur_best_cm = min(cur_best_cm, max_val)

    return cur_best_cm


# Sample usage and test:
# ----------------------
# v1 = 3 * np.sqrt(11) / 10
# v2 = 1 / 10
# S = np.array(
#     [
#         [v1, 0, -v1, 0, 0],
#         [0, v1, 0, -v1, 0],
#         [v2, v2, v2, v2, -1],
#     ]
# )
# res = basis_enumeration(S)
# print("Result: {}".format(res))
# print("True Solution: 0.1")
