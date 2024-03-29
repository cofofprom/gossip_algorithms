import matplotlib.pyplot as plt
import networkx

import gossip_algorithm
import numpy as np
from graphs import A_path, A_star, A_full, A_cell, A_connected, generate_random_graph
from benchmark import measure_algo, measure_convergency
import seaborn as sns
from const import N, ITER

# plt.plot(*measure_convergency(N, 500, 1000, 100))
# plt.show()

FLIERS = False

VALUES = np.array([[i] for i in range(N)])

simple_norms = measure_algo(gossip_algorithm.SimpleGossip, N, ITER, VALUES)
shift_norms = measure_algo(gossip_algorithm.ShiftRegister, N, ITER, VALUES)
jacobi_norms = measure_algo(gossip_algorithm.JacobiAlgo, N, ITER, VALUES)
# plt.subplot(1, 3, 1)
sns.boxplot(data=[simple_norms, shift_norms, jacobi_norms], showfliers=FLIERS)
plt.xticks([0, 1, 2], ['Simple gossip', 'Shift register', 'Jacobi polynomial'], rotation=-20)
# plt.title('Simple gossip')
# plt.subplot(1, 3, 2)
# sns.boxplot(y=shift_norms, color='b', showfliers=FLIERS)
# plt.title('Shift register')
# plt.subplot(1, 3, 3)
# sns.boxplot(y=jacobi_norms, color='r', showfliers=FLIERS)
# plt.title('Jacobi polynomial')
plt.show()
