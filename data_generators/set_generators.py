import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import special_ortho_group
from scipy.linalg import block_diag
import math

# 1 for .npy, 0 for json
SAVE_TYPE = 0


# Function to generate a permutation matrix
def generate_permutation_matrix(n):
    P = np.eye(n)
    np.random.shuffle(P.T)
    return P


def gen_min_can_pbasis(n, seed=None):
    B = np.concatenate((np.eye(n), -np.ones(n)[:, np.newaxis]), axis=1)
    if seed:
        np.random.seed(seed=seed)
        X = special_ortho_group.rvs(n)
        P = generate_permutation_matrix(B.shape[1])
        B = X @ B @ P
    B = normalize(B, axis=0)
    cm = 1 / np.sqrt(n**2 + 2 * (n - 1) * np.sqrt(n))
    return B, cm


def gen_min_uniform_set(n, seed=None):
    B = np.zeros((n, n + 1))

    # Define the ai values
    def ai(i, n):
        return np.sqrt((n - i + 1) * (n + 1) / (n * (n - i + 2)))

    # Fill the upper triangular part of M (excluding the last column)
    for i in range(n):
        B[i, i] = ai(i + 1, n)
        for j in range(i + 1, n):
            B[i, j] = -ai(i + 1, n) / (n - i)

    # Last column, copy previous row but negate last element
    B[:, n] = B[:, n - 1]
    B[n - 1, n] = -B[n - 1, n - 1]

    if seed:
        np.random.seed(seed=seed)
        X = special_ortho_group.rvs(n)
        P = generate_permutation_matrix(B.shape[1])
        B = X @ B @ P
    B = normalize(B, axis=0)
    cm = 1 / n
    return B, cm


def gen_min_delta_shift_pbasis(n, delta=0, seed=None):
    B = gen_min_uniform_set(n)[0]
    B[0, 1 : n + 1] += delta

    if seed:
        np.random.seed(seed=seed)
        X = special_ortho_group.rvs(n)
        P = generate_permutation_matrix(B.shape[1])
        B = X @ B @ P
    B = normalize(B, axis=0)
    cm = (1 - delta * n) / (np.sqrt(n * (delta**2 * n - 2 * delta + n)))
    return B, cm


# delta-shift pbasis
def gen_max_delta_shift_pbasis(n, delta=0, seed=None):
    B = np.eye(n) - np.ones((n, n)) * delta
    B = np.concatenate((B, -B), axis=1)

    if seed:
        np.random.seed(seed=seed)
        X = special_ortho_group.rvs(n)
        P = generate_permutation_matrix(B.shape[1])
        B = X @ B @ P
    B = normalize(B, axis=0)
    cm = (1 - delta * n) / (np.sqrt(n * (delta**2 * n - 2 * delta + 1)))
    return B, cm


# Augmented delta-shift pbasis
def gen_augmented_max_delta_shift_pbasis(n, delta=0, seed=None):
    B, solution = gen_max_delta_shift_pbasis(n, delta, None)
    gram = 1 / np.sqrt(n) * np.ones(n)

    counter = 0
    while counter < n**2:
        # Random sample from unit sphere
        v = np.random.normal(0, 1, n)
        v = v / np.linalg.norm(v)

        # Check if belongs to min cosine cone
        if v @ gram <= solution:
            B = np.column_stack((B, v))
            counter += 1
    if seed:
        np.random.seed(seed=seed)
        X = special_ortho_group.rvs(n)
        P = generate_permutation_matrix(B.shape[1])
        B = X @ B @ P
    B = normalize(B, axis=0)
    return B, solution


def gen_optimal_orthogonal_basis(n, s, seed=None):

    t = n / (s - n)
    r = n % (s - n)

    # Dimensions
    dim_1 = math.floor(t)
    dim_2 = math.ceil(t)
    cardinality_1 = s - n - r
    cardinality_2 = r

    min_pbasis_list_1 = [gen_min_uniform_set(dim_1)[0]] * cardinality_1
    min_pbasis_list_2 = [gen_min_uniform_set(dim_2)[0]] * cardinality_2

    combined_blocks = min_pbasis_list_1 + min_pbasis_list_2  # Concatenate both lists

    # Create the block diagonal matrix
    block_matrix = block_diag(*combined_blocks)

    solution = 1 / np.sqrt(cardinality_1 * dim_1**2 + cardinality_2 * dim_2**2)

    if seed:
        np.random.seed(seed=seed)
        X = special_ortho_group.rvs(n)
        P = generate_permutation_matrix(block_matrix.shape[1])
        block_matrix = X @ block_matrix @ P

    return block_matrix, solution


def gen_random_pspan_set(n, seed=None):
    if seed:
        np.random.seed(seed=seed)

    # Generate random invertible matrix (basis)
    B = np.random.rand(n, n)
    mx = np.sum(np.abs(B), axis=1)
    np.fill_diagonal(B, mx)
    coveredIndices = set()
    extensionSet = []

    # While all indices are not covered
    while len(coveredIndices) < n:
        # Random sample length and index
        length = np.random.randint(1, n // 2 + 2)
        startIndex = np.random.randint(n)
        # Check length exceeds target range
        if startIndex + length > n:
            continue

        # Create index subset and append to list
        indexSubset = list(range(startIndex, startIndex + length))
        if indexSubset not in extensionSet and indexSubset:
            extensionSet.append(indexSubset)
            for j in indexSubset:
                coveredIndices.add(j)

    # Construct new vectors and add to basis
    for indices in extensionSet:
        v = np.sum(-B[:, indices], axis=1)
        B = np.c_[B, v]
    B = normalize(B, axis=0)

    if seed:
        X = special_ortho_group.rvs(n)
        P = generate_permutation_matrix(B.shape[1])
        B = X @ B @ P

    return B, None


