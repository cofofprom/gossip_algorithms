import numpy as np
import random
from utils import qr_algo
import networkx


def createD(A):
    return np.array([[list(A[j]).count(1) if i == j else 0 for i in range(A.shape[0])] for j in range(A.shape[0])])


A_path = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0],
                   [1, 0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0, 1],
                   [0, 0, 0, 0, 0, 0, 0, 1, 0]])

D_path = createD(A_path)

A_star = np.array([
    [0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
])

D_star = createD(A_star)

A_full = np.array([
    [0 if i == j else 1 for i in range(9)] for j in range(9)
])

D_full = createD(A_full)

A_cell = np.array([
    [0, 1, 0, 1, 0, 0, 0, 0, 0],  # 1 -> 2, 4
    [1, 0, 1, 0, 1, 0, 0, 0, 0],  # 2 -> 1, 3, 5
    [0, 1, 0, 0, 0, 1, 0, 0, 0],  # 3 -> 2, 6
    [1, 0, 0, 0, 1, 0, 1, 0, 0],  # 4 -> 1, 5, 7
    [0, 1, 0, 1, 0, 1, 0, 1, 0],  # 5 -> 2, 4, 6, 8
    [0, 0, 1, 0, 1, 0, 0, 0, 1],  # 6 -> 3, 5, 9
    [0, 0, 0, 1, 0, 0, 0, 1, 0],  # 7 -> 4, 8
    [0, 0, 0, 0, 1, 0, 1, 0, 1],  # 8 -> 5, 7, 9
    [0, 0, 0, 0, 0, 1, 0, 1, 0]  # 9 -> 6, 8
])

D_cell = createD(A_cell)

A_connected = np.array([
    [0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 0, 0],
    [1, 1, 1, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 1, 1, 1, 0]
])

D_connected = createD(A_connected)


def generate_random_graph(n, p=0.5):
    g = networkx.erdos_renyi_graph(n, p)
    while not networkx.is_connected(g):
        g = networkx.erdos_renyi_graph(n, p)
    return g, networkx.to_numpy_array(g)
