import daal4py as d4p
import numpy as np


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
    #eigenvalues = [curr[x, x] for x in range(matrix.shape[0])]
    return eigenvalues
