import json
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import special_ortho_group
from data_generators.set_generators import generate_permutation_matrix

import importlib

# Benchmarking Settings
# ---------------------
# 1 for npy 0 for json
FILE_TYPE = 0
SEED = 4242
CHECKPOINT_INTERVAL = 1
TOTAL_ROTATIONS = 3

# Comment out methods you don't want to test
methods = {
    "QP Gurobi": {
        "module": importlib.import_module("algorithms.branch_and_bound_QP_QCLP"),
        "function": "solve_QP",
        "runs": 1,
        "args": {"threads": 1},
    },
    "QP Gurobi Multithread": {
        "module": importlib.import_module("algorithms.branch_and_bound_QP_QCLP"),
        "function": "solve_QP",
        "runs": 1,
        "args": {"threads": 8},
    },
    "Jarry-Bolduc 2020": {
        "module": importlib.import_module("algorithms.basis_enumeration"),
        "function": "basis_enumeration",
        "runs": 1,
    },
    "Regis KKT": {
        "module": importlib.import_module("algorithms.kkt_enumeration"),
        "function": "kkt_enumeration",
        "runs": 1,
    },
    "QCLP Gurobi": {
        "module": importlib.import_module("algorithms.branch_and_bound_QP_QCLP"),
        "function": "solve_QCLP",
        "runs": 1,
        "args": {"threads": 1},
    },
    "QCLP Gurobi Multithread": {
        "module": importlib.import_module("algorithms.branch_and_bound_QP_QCLP"),
        "function": "solve_QCLP",
        "runs": 1,
        "args": {"threads": 8},
    },
    "Random Linear Program": {
        "module": importlib.import_module("algorithms.random_linear_program"),
        "function": "random_lp_method",
        "runs": 4,
    },
    "Random Linear Program Multithread": {
        "module": importlib.import_module("algorithms.random_linear_program"),
        "function": "random_lp_method_mt",
        "runs": 4,
        "args": {"threads": 8},
    },
    "LRS Vertex Enumeration Rational": {
        "module": importlib.import_module("algorithms.vertex_enumeration"),
        "class": getattr(
            importlib.import_module("algorithms.vertex_enumeration"), "LRS"
        ),
        "function": "vertex_enumeration",
        "pre-function": "set_initial_dictionary",
        "runs": 1,
    },
}


# Load test cases from JSON files
def load_test_cases(test_folder):
    test_cases = []
    for root, dirs, files in os.walk(test_folder):
        for file in files:
            if file.endswith(".json"):
                relative_path = os.path.relpath(root, test_folder)
                file_path = os.path.join(root, file)

                # Combine subdirectory and filename to form the test case name
                test_name = os.path.join(relative_path, file).replace(
                    ".json", ""
                )  # Remove .json
                with open(file_path, "r") as f:
                    test_case = json.load(f)
                    test_cases.append(
                        {
                            "name": test_name,
                            "matrix": np.array(test_case["matrix"]),
                            "actual_solution": np.array(test_case["solution"]),
                        }
                    )
    return test_cases


# Format numpy arrays, lists, or single values into strings
def format_solution(sol):
    if not sol:
        return "None"
    elif isinstance(sol, np.ndarray) and sol.ndim == 0:
        return f"{sol:.6f}"
    elif isinstance(sol, (list, np.ndarray)):
        return ", ".join([f"{x:.6f}" for x in np.round(sol, 6)])
    else:
        return f"{sol:.6f}"


# Main benchmarking function
def benchmark_methods(test_cases, timeout, tolerance=1e-6, output_file="results.csv"):
    results = []

    # Loop over the test cases
    np.random.seed(seed=SEED)
    for test_idx, test_case in enumerate(test_cases):

        # Load test
        test_name = test_case["name"]
        actual_solution = test_case["actual_solution"]
        A = test_case["matrix"]

        # Random rotation and permutation
        for num_rotation in range(TOTAL_ROTATIONS):
            R = special_ortho_group.rvs(A.shape[0])
            P = generate_permutation_matrix(A.shape[1])
            A = R @ A @ P

            # Logging
            print(
                "Currently running test: {}, Rotation: {}".format(
                    test_name, num_rotation
                )
            )

            # Store the the current test case and known solution
            test_results = {
                "Test Set": test_name + f"/rotation_{num_rotation}",
                "Actual Solution": format_solution(actual_solution),
            }

            # Save RNG state
            rng_state = np.random.get_state()

            # Run methods
            for method_name, method_info in methods.items():

                # Get method information
                num_runs = method_info["runs"]
                args = {}
                if "args" in method_info.keys():
                    args = method_info["args"]

                # Grabs the correct method
                if method_info.get("class"):
                    class_params = {}
                    class_params["A"] = A
                    class_params["b"] = np.ones(A.shape[1])
                    class_obj = method_info["class"](**class_params)
                    pre_method = getattr(class_obj, method_info["pre-function"])
                    method = getattr(class_obj, method_info["function"])
                else:
                    pre_method = None
                    method = getattr(method_info["module"], method_info["function"])

                # Run method
                for run in range(num_runs):
                    elapsed_time = 0
                    print("Run number: {}".format(run))
                    try:
                        # Call the method with timeout
                        if pre_method:
                            elapsed_time = pre_method(timeout=timeout)
                        # Check if premethod exceeds time
                        if elapsed_time >= timeout:
                            solution = None
                        else:
                            new_time = timeout - elapsed_time
                            solution = method(
                                A,
                                timeout=new_time,
                                known_sol=actual_solution,
                                tol=tolerance,
                                **args,
                            )
                    except Exception as e:
                        solution = None
                        print(f"{method_name} failed on {test_name} due to {str(e)}")

                    # Store the result for this run
                    test_results[f"{method_name}_run_{run+1}"] = format_solution(
                        solution
                    )

            # Load RNG state after randomized methods
            np.random.set_state(rng_state)

            # Append test results
            results.append(test_results)

        # Checkpoint: Save every `CHECKPOINT_INTERVAL` tests
        if (test_idx + 1) % CHECKPOINT_INTERVAL == 0:
            results_df = pd.DataFrame(results)

            # Append to CSV, only write header if the file doesn't exist
            try:
                with open(output_file, "r") as f:
                    file_exists = True
            except FileNotFoundError:
                file_exists = False

            results_df.to_csv(
                output_file, mode="a", header=not file_exists, index=False
            )
            print(f"Checkpoint saved after {test_idx + 1} tests to {output_file}")

            # Clear results list to free memory
            results = []

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, mode="a", header=not file_exists, index=False)
        print(f"Final checkpoint saved to {output_file}")


if __name__ == "__main__":
    # Path of test folder
    test_folder = "test_cases"
    test_cases = load_test_cases(test_folder)
    benchmark_methods(test_cases, timeout=30, output_file="results.csv")
