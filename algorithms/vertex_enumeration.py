import numpy as np
from fractions import Fraction
from scipy.optimize import linprog
from copy import deepcopy
import time


class LRS:
    def __init__(self, A, b):
        self.A = A.T
        self.b = b
        self.d, self.m = A.shape

        self.B = list(range(self.m + 1))
        self.N = list(range(self.m + 1, self.m + self.d + 1))
        self.dictionary = None

    def set_initial_dictionary(self, timeout=None):
        start_time = time.time()
        n = self.d
        tol = 1e-8

        # Find initial basis by solving LP
        c = np.ones(n)
        bounds = [(None, None) for _ in range(self.d)]
        res = linprog(c, A_ub=self.A, b_ub=self.b, bounds=bounds).x

        # Find index of tight constraints
        diff = self.A @ res - self.b
        zero_indices = np.where(np.isclose(diff, 0, atol=tol))[0]

        # If initial basis is degenerate grab last n rows
        zero_indices = zero_indices[:n]

        # Reorder the matrix A to have inequalities at the bottom
        rows_to_move = self.A[zero_indices, :]
        remaining_rows = np.delete(self.A, zero_indices, axis=0)
        self.A = np.vstack((remaining_rows, rows_to_move))

        entries_to_move = self.b[zero_indices]
        remaining_rows = np.delete(self.b, zero_indices)
        self.b = np.concatenate((remaining_rows, entries_to_move))

        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
            print("Initial dictionary - LP - timeout")
            return elapsed_time
        # Rewrite constraints
        I = np.eye(self.m)
        A_bar = np.hstack((self.A, I))

        A_B = A_bar[:, 0 : self.m]
        A_N = A_bar[:, -self.d :]

        # Test floating inversion
        A_B_inv = np.linalg.inv(A_B)
        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
            print("Initial dictionary - matrix inversion - timeout")
            return elapsed_time

        # Perform the matrix multiplication and construct the dictionary
        D = np.hstack((A_B_inv @ A_N, A_B_inv @ self.b[:, np.newaxis]))

        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
            print("Initial dictionary - matrix multiplication - timeout")
            return elapsed_time

        # Construct cost row
        cost_row = np.array([1 for _ in range(self.d)] + [0])
        D = np.vstack((cost_row, D))
        I = np.eye(self.m + 1)
        D = np.hstack((I, D))
        D = self.array_to_fraction(D)
        elapsed_time = time.time() - start_time
        if timeout is not None and elapsed_time >= timeout:
            print("Initial dictionary - fraction conversion - timeout")
            return elapsed_time
        self.dictionary = D
        return elapsed_time

    def pivot(self, r, s):
        B_index = self.B.index(r)
        N_index = self.N.index(s)

        pivot_element = self.dictionary[B_index, s]
        self.dictionary[B_index] /= pivot_element

        for i in range(self.m + 1):
            if i != B_index:
                factor = self.dictionary[i, s]
                self.dictionary[i] -= factor * self.dictionary[B_index]

        self.B[B_index], self.N[N_index] = s, r

    def selectpivot(self):
        min_s = None
        min_j = None
        for j, s in enumerate(self.N):
            if (min_s is None or s < min_s) and self.dictionary[0, s] > 0:
                min_s = s
                min_j = j

        r = self.lexminratio(min_s)
        if r != 0:
            return r, min_j
        return None, None

    def lexminratio(self, s):
        # -1 if v1 < v2 lexicographically, 0 if equal, 1 if v1 > v2
        def lexicographic_compare(v1, v2):
            for a, b in zip(v1, v2):
                if a != b:
                    return np.sign(a - b)
            return np.sign(len(v1) - len(v2))

        # If empty set to 0
        min_ratio = None
        min_index = 0
        D_nonempty_flag = False

        for i in range(self.d + 1, self.m + 1):
            if self.dictionary[i, s] > 0:
                ratio = self.dictionary[i, [-1] + self.B] / self.dictionary[i, s]
                if min_ratio is None or lexicographic_compare(ratio, min_ratio) < 0:
                    min_index = i
                    min_ratio = ratio
                    D_nonempty_flag = True

        if D_nonempty_flag:
            return self.B[min_index]
        return 0

    def lexmin(self, s):
        for i in range(self.m + 1):
            for j in range(self.d):
                if (
                    self.B[i] > self.N[j]
                    and self.dictionary[i, -1] == 0
                    and self.dictionary[i, j] != 0
                ):
                    return False
        return True

    def reverse(self, v):
        if self.dictionary[0, v] <= 0:
            return False, None

        u = self.lexminratio(v)
        i = self.B.index(u)
        if u == 0:
            return False, None

        w_bar = (
            self.dictionary[0, :]
            - (self.dictionary[0, v] / self.dictionary[i, v]) * self.dictionary[i, :]
        )

        for index, j in enumerate(self.N):
            if j < u and w_bar[j] < 0:
                return False, None

        return True, u

    def vertex_enumeration(self, D, timeout=None, known_sol=None, tol=0):
        start_time = time.time()
        B_STAR = np.array(list(range(self.m + 1)))
        cache = []
        j = 0
        max_norm_squared = self.norm_squared(self.get_vertex())
        if self.lexmin(0):
            v = self.get_vertex()
        counter = 0
        while True:
            while j < self.d:
                if timeout is not None and time.time() - start_time >= timeout:
                    return 1 / np.sqrt(float(max_norm_squared))
                if (
                    known_sol
                    and abs(1 / np.sqrt(float(max_norm_squared)) - known_sol) < tol
                ):
                    return 1 / np.sqrt(float(max_norm_squared))
                v = self.N[j]
                reverse_possible, u = self.reverse(v)
                if reverse_possible:
                    cache.append(
                        [
                            deepcopy(self.dictionary),
                            deepcopy(self.B),
                            deepcopy(self.N),
                            j,
                        ]
                    )
                    self.pivot(u, v)
                    if self.lexmin(0):
                        v = self.get_vertex()
                        candidate_norm_squared = self.norm_squared(self.get_vertex())
                        if candidate_norm_squared > max_norm_squared:
                            max_norm_squared = candidate_norm_squared
                    j = 0
                else:
                    j += 1

            counter += 1
            self.dictionary, self.B, self.N, j = cache.pop()
            j += 1
            if j >= self.d and np.all(self.B == B_STAR):
                break
        res = 1 / np.sqrt(float(max_norm_squared))
        return res

    def get_vertex(self):
        return self.dictionary[1 : self.d + 1, -1]

    @staticmethod
    def array_to_fraction(arr):
        fraction_array = np.vectorize(Fraction)(arr)
        return fraction_array

    @staticmethod
    def norm_squared(v):
        return sum(x**2 for x in v)


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
# lrs = LRS(S, np.ones(S.shape[1]))
# lrs.set_initial_dictionary()
# res = lrs.vertex_enumeration(lrs.dictionary)
# print("Result: {}".format(res))
# print("True Solution: 0.1")
