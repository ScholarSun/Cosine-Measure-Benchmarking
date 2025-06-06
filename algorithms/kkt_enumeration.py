import numpy as np
from sklearn.preprocessing import normalize
import itertools
from scipy.optimize import linprog
import time

EPSILON = 1e-12


# Checks if K^circ (T, x) intersect with S is nonempty
def is_active_subset(fullSet, subset_indices, x, full_active_set=False):
    candidate_cm = fullSet[:, subset_indices[0]] @ x
    cosines = fullSet.T @ x
    if full_active_set:
        cosines = np.delete(cosines, subset_indices)
    res = np.all(cosines <= candidate_cm + EPSILON)
    return res


# Checks if 0 is in the convex hull of S
def zero_in_conv(S):
    n, d = S.shape

    b = np.zeros(n)
    b[-1] = 1
    c = np.zeros(d)

    bounds = [(0, None) for i in range(d)]
    res = linprog(c, A_eq=S, b_eq=b, bounds=bounds).status

    if res == 0:
        return True
    elif res == 2:
        return False
    else:
        raise Exception("Linear program is neither infeasible nor optimal")


# Solves for a vector x in the intersection of pspan(S) and Theta(S)
def solve_pspan_intersect_uniform(S):
    n, k = S.shape
    D = np.zeros((n, k - 1))
    b = np.zeros(k - 1)
    c = np.ones(k)

    # Note: this can be more simple in matrix multiplication form
    for i in range(k - 1):
        D[:, i] = S[:, i + 1] - S[:, 0]
    A = D.T @ S

    bounds = [(0, 1) for i in range(k)]

    res = linprog(-c, A_eq=A, b_eq=b, bounds=bounds).x
    return res


def select_orthogonal_vector(S):
    u, s, vh = np.linalg.svd(S.T)
    null_space = np.compress(s <= EPSILON, vh, axis=0)
    return null_space[0]


# Input: arbitrary n x d set of vectors where n > 1
def kkt_enumeration(S, timeout=None, known_sol=None, tol=1e-6):
    # Normalize and remove redundant column
    S = normalize(S, axis=0)
    S = np.unique(S, axis=1)

    # New dimemsions
    n, d = S.shape

    start_time = time.time() if timeout is not None else None
    min_cm = 100

    # For each subset of S
    for subset_size in range(2, n + 1):
        for indices, subset_indices in enumerate(
            itertools.combinations(range(d), subset_size)
        ):

            # Subset construction
            sub = S[:, subset_indices]
            sub_aff = np.vstack((sub, np.ones(subset_size)))

            # If time out exceeded
            if timeout is not None and time.time() - start_time >= timeout:
                break
            # If close enough to known solution
            if known_sol and abs(min_cm - known_sol) < tol:
                break

            # Check if subset is linearly independent
            if np.linalg.matrix_rank(sub) == subset_size:
                beta_coef = solve_pspan_intersect_uniform(sub)
                x = sub @ beta_coef
                if np.linalg.norm(x) == 0:
                    continue
                x = x / np.linalg.norm(x)
                if is_active_subset(S, subset_indices, x):
                    cur_cm = x @ sub[:, 0]
                    if cur_cm < min_cm:
                        min_cm = cur_cm

                if is_active_subset(S, subset_indices, -x):
                    cur_cm = -x @ sub[:, 0]
                    if cur_cm < min_cm:
                        min_cm = cur_cm

            # Check if subset is affinely independent
            elif np.linalg.matrix_rank(sub_aff) == subset_size and zero_in_conv(
                sub_aff
            ):
                x = select_orthogonal_vector(sub)
                if is_active_subset(S, subset_indices, x):
                    cur_cm = x @ sub[:, 0]
                    if cur_cm < min_cm:
                        min_cm = cur_cm
                if is_active_subset(S, subset_indices, -x):
                    cur_cm = -x @ sub[:, 0]
                    if cur_cm < min_cm:
                        min_cm = cur_cm
            else:
                continue

    return min_cm


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
# res = kkt_enumeration(S)
# print("Result: {}".format(res))
# print("True Solution: 0.1")
