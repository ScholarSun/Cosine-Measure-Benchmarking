import numpy as np
import gurobipy as gp
from scipy import optimize
import multiprocessing
import time


def sample_unit_vector_uniform(n):
    vec = np.random.normal(0, 1, n)
    return vec / np.linalg.norm(vec)


def solve_lp(A, c):
    n, m = A.shape
    bounds = [(None, None) for _ in range(n)]
    b = np.ones(m)
    res = optimize.linprog(c, A_ub=A.T, b_ub=b, bounds=bounds)

    if res.success:
        return res.x, res.fun
    else:
        return None, None


def solve_lp_gurobi(A, c):
    n, m = A.shape
    model = gp.Model()
    model.setParam("OutputFlag", 0)

    # Variables
    x = model.addMVar(n, lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="x")

    # Solve
    model.setObjective(c @ x, gp.GRB.MAXIMIZE)
    model.addConstr(A.T @ x <= 1)
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        return x.X, model.ObjVal
    else:
        return None, None


# Single-threaded Random LP method
def random_lp_method(A, iters=None, timeout=None, known_sol=None, tol=1e-6):
    n, m = A.shape  # Number of dimensions
    max_norm = -100

    if iters is None and timeout is None:
        raise ValueError("You must specify either iters or timeout")

    start_time = time.time() if timeout is not None else None
    current_iter = 0

    # Loop until either iterations or timeout is reached
    while True:
        if iters is not None and current_iter >= iters:
            break
        if timeout is not None and time.time() - start_time >= timeout:
            break
        if known_sol and abs(1 / max_norm - known_sol) < tol:
            break
        c = sample_unit_vector_uniform(n)

        # Solve the linear program for the sampled direction
        x, _ = solve_lp_gurobi(A, -c)

        # Update the maximum norm if solution exists and has a larger norm
        cur_norm = np.linalg.norm(x)
        if cur_norm > max_norm:
            max_norm = cur_norm

        current_iter += 1
    return 1 / max_norm if max_norm != -100 else None


def worker(A, time_limit, max_value, start_time, tol, opt_norm=None):
    start_time = time.time()
    while time.time() - start_time < time_limit:
        c = sample_unit_vector_uniform(A.shape[0])
        result = solve_lp_gurobi(A, -c)
        with max_value.get_lock():
            max_value.value = max(max_value.value, np.linalg.norm(result[0]))
            if opt_norm and abs(max_value.value - opt_norm) < tol:
                break


# Multithreaded Random LP method
def random_lp_method_mt(A, timeout=None, known_sol=None, tol=1e-6, threads=None):

    if timeout is None:
        raise ValueError("You must specify timeout")

    opt_norm = None
    if known_sol:
        opt_norm = 1 / known_sol

    max_value = multiprocessing.Value("d", float("-inf"))
    start_time = time.time()

    processes = [
        multiprocessing.Process(
            target=worker, args=(A, timeout, max_value, start_time, tol, opt_norm)
        )
        for _ in range(threads or multiprocessing.cpu_count())
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    return 1 / max_value.value


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

# print("True Solution: 0.1")
# res_lp = random_lp_method(S, timeout=1)
# print("Result non-MT: {}".format(res_lp))

# # Guard necessary for multiprocessing
# if __name__ == "__main__":
#     res_lp_mt = random_lp_method_mt(S, timeout=4, threads=2)
#     print("Result MT: {}".format(res_lp_mt))


