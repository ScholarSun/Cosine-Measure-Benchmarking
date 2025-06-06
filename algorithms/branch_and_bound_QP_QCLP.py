import gurobipy as gp
from gurobipy import GRB
import numpy as np


def solve_QCLP(S, timeout=None, known_sol=None, tol=None, threads=None):
    n, k = S.shape
    m = gp.Model("CMP-QCLP")

    # Other Model Options
    # m.setParam('OutputFlag', 0)
    # m.setParam("MIPFocus", 1)

    if threads:
        m.setParam("Threads", threads)

    if timeout:
        m.setParam("TimeLimit", timeout)

    vars = list(range(n))

    # Variables
    for i in range(n):
        vars[i] = m.addVar(name="x" + str(i + 1), lb=-1, ub=1)

    # lb 0 for pspanning set
    z = m.addVar(name="z", lb=0, ub=1)
    m.setObjective(z, GRB.MINIMIZE)

    # Add linear constraints
    for i in range(k):
        m.addConstr(vars @ S.T[i] <= z, "c" + str(i))

    # Unit circle constraint
    m.addConstr(gp.quicksum(x**2 for x in vars) == 1, "unit-circle")
    m.optimize()

    cm = m.ObjVal
    return cm


def solve_QP(S, timeout=None, known_sol=None, tol=None, threads=None):
    n, k = S.shape
    m = gp.Model("CMP-QP")

    # Other Model Options
    # m.setParam('OutputFlag', 0)
    # m.setParam("MIPFocus", 1)

    if threads:
        m.setParam("Threads", threads)

    if timeout:
        m.setParam("TimeLimit", timeout)

    # Variables
    x = m.addMVar(n, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
    m.addConstr(x @ S <= 1)

    m.setObjective(x @ x, GRB.MAXIMIZE)
    m.optimize()
    cm = 1 / np.sqrt(m.ObjVal)
    return cm


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
# res_QCLP = solve_QCLP(S)
# res_QP = solve_QP(S)
# print("Result QCLP: {}".format(res_QCLP))
# print("Result QP: {}".format(res_QP))
# print("True Solution: 0.1")
