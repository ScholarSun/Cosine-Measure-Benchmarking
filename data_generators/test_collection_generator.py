import numpy as np
from sklearn.preprocessing import normalize
from scipy.stats import special_ortho_group
from set_generators import *
import json
import os

# 1 for .npy, 0 for json
SAVE_TYPE = 0

def save_matrix_and_solution(matrix, solution, filename):
    if SAVE_TYPE == 0:
        matrix = matrix.tolist()

    data = {"matrix": matrix, "solution": solution}
    if SAVE_TYPE == 0:
        with open(filename + ".json", "w") as f:
            json.dump(data, f, indent=4)
    elif SAVE_TYPE == 1:
        np.save(filename, data)


# Define different delta functions
def delta_zero(n):
    return 0


def delta_1_2(n):
    return 1 / (2 * n)


def delta_2_3(n):
    return 2 / (3 * n)


# List of delta functions
delta_functions = [("δ_0", delta_zero), ("δ_1_2n", delta_1_2), ("δ_1_3n", delta_2_3)]


def gen_data_instance(
    n,
    func,
    cur_dir,
    use_delta,
    is_intermediate,
    num_tests,
    delta=None,
    int_size=None,
    seed=None,
):
    if seed:
        np.random.seed(seed=seed)
    for test_num in range(num_tests):

        if use_delta:
            matrix, solution = func(n, delta)
        elif is_intermediate:
            matrix, solution = func(n, int_size)
        else:
            matrix, solution = func(n)

        filename = os.path.join(cur_dir, "test_{}".format(test_num + 1))
        save_matrix_and_solution(matrix, solution, filename)

        print(f"Saved test case {filename}")

    return


# Function to generate and save test sets for different n values
def generate_test_set(output_dir, n_values, seed=None):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate various test cases
    # (test name, function, use delta, intermediate?, number of tests, number of transformations)
    # number of transformation is unused
    test_cases = [
        ("min_can_pbasis", gen_min_can_pbasis, False, False, 1, 5),
        ("min_pbasis", gen_min_delta_shift_pbasis, True, False, 1, 5),
        ("max_pbasis", gen_max_delta_shift_pbasis, True, False, 1, 5),
        ("optimal_orthogonal", gen_optimal_orthogonal_basis, False, True, 1, 5),
        ("augmented_max_pbasis", gen_augmented_max_delta_shift_pbasis, True, False, 3, 3),
        ("random_pspan", gen_random_pspan_set, False, False, 3, 3),
    ]

    # For each test case, create a separate folder
    for (
        name,
        func,
        uses_delta,
        is_intermediate,
        num_tests,
        num_transformations,
    ) in test_cases:
        test_case_dir = os.path.join(output_dir, name)
        if not os.path.exists(test_case_dir):
            os.makedirs(test_case_dir)

        # For each dimension make a directory
        for n in n_values:
            dimension_dir = os.path.join(test_case_dir, "n={}".format(n))
            if not os.path.exists(dimension_dir):
                os.makedirs(dimension_dir)

            # For each modifier make a directory
            if uses_delta:
                for delta_name, delta_func in delta_functions:
                    delta_dir = os.path.join(dimension_dir, delta_name)
                    if not os.path.exists(delta_dir):
                        os.makedirs(delta_dir)
                    delta = delta_func(n)
                    gen_data_instance(
                        n,
                        func,
                        delta_dir,
                        True,
                        False,
                        num_tests,
                        delta=delta,
                        seed=seed,
                    )

            elif is_intermediate:
                s_min = n + 1
                s_max = 2 * n
                s_middle = (s_min + s_max) / 2

                # Calculate the evenly spaced values for s centered in the middle, excluding endpoints
                half_interval = (
                    s_max - s_min
                ) / 4  # Half of the total range between s_min and s_max

                # Generate max_sub_tests values evenly spaced around the center, excluding extremes
                s_values = np.linspace(
                    s_middle - half_interval, s_middle + half_interval, 2, dtype=int
                )

                for s in s_values:
                    s_dir = os.path.join(dimension_dir, "s={}".format(s))
                    if not os.path.exists(s_dir):
                        os.makedirs(s_dir)
                    gen_data_instance(
                        n,
                        func,
                        s_dir,
                        False,
                        True,
                        num_tests,
                        int_size=s,
                        seed=seed,
                    )
            else:
                gen_data_instance(
                    n,
                    func,
                    dimension_dir,
                    False,
                    False,
                    num_tests,
                    seed=seed,
                )


# Sample usage:
# n_values = [10, 13, 15, 18, 21, 24, 27, 30, 40, 50, 60, 70, 80, 90, 100]
# generate_test_set(output_dir='path/to/test/directory/test_cases', n_values=n_values, seed=4242)
