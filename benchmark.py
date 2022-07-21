import matplotlib.pyplot as plt
import networkx

from gossip_algorithm import SimpleGossip, ShiftRegister, JacobiAlgo, GossipAlgo
from graphs import generate_random_graph
from numpy.linalg import norm
import numpy as np
from utils import calculate_spectral
from const import TESTS_NUMBER


def measure_algo(algo_class, n, iter, values):
    norms = []
    for _ in range(TESTS_NUMBER):
        g, graph = generate_random_graph(n)
        algo = algo_class(values, iter, graph)
        result = algo.compute()
        norms.append(norm(result - algo.means))

    return norms


def measure_convergency(n, r, max, skip):
    g, mat = generate_random_graph(n)
    w = GossipAlgo(np.array([[1] for i in range(0, max, skip)]), 1, mat).matrix_W
    t_array = [i for i in range(1, max)]
    d_array = []
    for t in t_array:
        d_array.append(calculate_spectral(t, r, w))

    return t_array, d_array
