import daal4py as d4p
import numpy as np
import random


def qr_algo(matrix, precision):
    algo = d4p.qr()
    curr = matrix
    res = algo.compute(curr)
    q, r = res.matrixQ, res.matrixR
    curr = r.dot(q)
    last = np.array([curr[x, x] for x in range(matrix.shape[0])])
    while True:
        res = algo.compute(curr)
        q, r = res.matrixQ, res.matrixR
        curr = r.dot(q)
        eigenvalues = np.array([curr[x, x] for x in range(matrix.shape[0])])
        if np.linalg.norm(eigenvalues - last) < 10 ** (-precision):
            return eigenvalues
        last = eigenvalues
    # eigenvalues = [curr[x, x] for x in range(matrix.shape[0])]
    return eigenvalues


def calculate_spectral(t, r, w):
    lazy_W = (np.eye(w.shape[0]) + w) * 0.5

    avg_init_vertex_count = 0

    for _ in range(r):
        curr_vertex = 0
        for step in range(t):
            # print(f"DEBUG: current vertex is {curr_vertex}")
            chance = random.random()
            lower = 0
            for candidate in range(lazy_W.shape[0]):
                borders = (lower, lower + lazy_W[curr_vertex, candidate])
                if borders[0] <= chance < borders[1]:
                    curr_vertex = candidate
                    break
                else:
                    lower += lazy_W[curr_vertex, candidate]

        if curr_vertex == 0:
            avg_init_vertex_count += 1

    p_t = avg_init_vertex_count / r
    return (-2) * (np.log(p_t / t + 0.00001))
    # return (-2) * (np.log(p_t) / np.log(self.T_LIM))
