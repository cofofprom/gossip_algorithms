import matplotlib.pyplot as plt

from gossip_algorithm import SimpleGossip, ShiftRegister, JacobiAlgo
from graphs import generate_random_graph
from numpy.linalg import norm

TESTS_NUMBER = 100


def measure_algo(algo_class, n, iter, values):
    norms = []
    for _ in range(TESTS_NUMBER):
        g, graph = generate_random_graph(n)
        algo = algo_class(values, iter, graph)
        result = algo.compute()
        norms.append(norm(result - algo.means))

    return norms