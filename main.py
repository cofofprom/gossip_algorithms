# import math
# import random
#
# from sklearnex import patch_sklearn
#
# patch_sklearn()
# import numpy as np
# import daal4py as d4p
# from graphs import A_path, A_star, A_full, A_cell, A_connected, createD
# from matplotlib import pyplot as plt
# import random
#
#
# def qr_algo(W, iter):
#     algo = d4p.qr()
#     curr = W
#     for x in range(iter):
#         res = algo.compute(curr)
#         q, r = res.matrixQ, res.matrixR
#         curr = r.dot(q)
#         #print([curr[x, x] for x in range(W.shape[0])])
#     eigenvalues = [curr[x, x] for x in range(W.shape[0])]
#     return eigenvalues
#
#
# def calculate_W(a):
#     d = createD(a)
#     return np.eye(9) + (a - d) * 1 / np.max(d)
#
#
# def simple_gossip(n, W, values, efficiency_array, means):
#     for t in range(1, n):
#         values = W.dot(values)
#         diff = values - means
#         norm = np.linalg.norm(diff)
#         print(norm)
#         efficiency = np.exp(np.log(norm) / t)
#         efficiency_array.append(efficiency)
#     print(efficiency_array[0: 5])
#     return values
#
#
# def jacobi_polynomial(n, W, values, d, efficiency_array, means):
#     a_0 = (d + 4) / (2 * (2 + d))
#     b_0 = d / (2 * (2 + d))
#     prev = values
#     curr = a_0 * W.dot(values) + b_0 * values
#     efficiency = np.linalg.norm(curr - means)
#     efficiency_array.append(efficiency)
#     for t in range(2, n):
#         a = ((2 * t + (d / 2) + 1) * (2 * t + (d / 2) + 2)) / (2 * ((t + 1 + (d / 2)) ** 2))
#         b = ((d ** 2) * (2 * t + (d / 2) + 1)) / (8 * ((t + 1 + (d / 2)) ** 2) * (2 * t + (d / 2)))
#         c = ((t ** 2) * (2 * t + (d / 2) + 2)) / (((t + 1 + (d / 2)) ** 2) * (2 * t + (d / 2)))
#
#         iter_value = a * W.dot(curr) + b * curr - c * prev
#
#         curr, prev = iter_value, curr
#
#         efficiency = np.linalg.norm(curr - means)
#         efficiency_array.append(np.exp(np.log(efficiency) / t))
#
#     return curr
#
#
# def shift_register(n, W, values, gamma, efficiency_array, means):
#     omega = 2 * (1 - math.sqrt(gamma * (1 - (gamma / 4)))) / ((1 - (gamma / 2)) ** 2)
#     curr, prev = W.dot(values), values
#     efficiency = np.linalg.norm(curr - means)
#     efficiency_array.append(efficiency)
#     for t in range(2, n):
#         iter_values = omega * W.dot(curr) + (1 - omega) * prev
#         curr, prev = iter_values, curr
#         efficiency = np.linalg.norm(curr - means)
#         print(efficiency)
#         efficiency_array.append(np.exp(np.log(efficiency) / t))
#     return curr
#
#
# VALUES = np.array([[random.random()] for i in range(9)])
# mean = np.mean(VALUES)
# MEANS = np.array([[mean] for i in range(9)])
# print(VALUES.T)
#
# W = calculate_W(A_cell)
# print('W =\n', W)
#
# #eigenvalues = np.linalg.eig(W)[0]
# eigenvalues = qr_algo(W, 10000) # контроль ошибки
# eigenvalues.sort()
# print("eigenvalues", eigenvalues)
# gamma = 1 - eigenvalues[-2]
# abs_gamma = min(gamma, eigenvalues[0] + 1)
# print('gamma =', gamma)
# print('absolute gamma =', abs_gamma)
# omega = 2 * (1 - math.sqrt(gamma * (1 - (gamma / 4)))) / ((1 - (gamma / 2)) ** 2)
# #omega_measure = 1 - 2 * (math.sqrt(gamma * (1 - (gamma / 4))) - (gamma / 2)) / (1 - gamma)
# omega_measure = 1 + (1/9) * ((4 - np.sqrt(4 - (1 + (eigenvalues[-2]) ** 2)))/(1 + eigenvalues[-2]) - 2)
# #omega_measure = 1 - math.sqrt(gamma)
# print('omega = ', omega)
# print('omega_measure = ', omega_measure)
#
# ITER_NUM = 201
#
# X = list(range(1, ITER_NUM))
# Y = []
#
# measure = omega_measure
#
# VALUES = jacobi_polynomial(ITER_NUM, W, VALUES, 88, Y, MEANS)
# #VALUES = shift_register(ITER_NUM, W, VALUES, gamma, Y, MEANS)
# # VALUES = simple_gossip(1001, W, VALUES, Y, MEANS)
# print("Value =", VALUES)
# print("measure:", measure)
# plt.plot(X, Y)
# plt.plot(X, [(measure) for x in X])
# plt.show()
import matplotlib.pyplot as plt
import networkx

import gossip_algorithm
import numpy as np
from graphs import A_path, A_star, A_full, A_cell, A_connected, generate_random_graph
from benchmark import measure_algo
import seaborn as sns

# g, graph = generate_random_graph(9)
# algo = gossip_algorithm.SimpleGossip(VALUES, 151, A_path)
# algo = gossip_algorithm.ShiftRegister(VALUES, 251, A_path)
# algo = gossip_algorithm.JacobiAlgo(VALUES, 301, A_path)
# print(algo.compute())

# plt.subplot(1, 2, 1)
# plt.plot(algo.x_list, algo.y_list)
# plt.plot(algo.x_list, [algo.measure for x in algo.x_list])
# plt.subplot(1, 2, 2)
# networkx.draw(g)
# plt.show()

N = 20
ITER = 25
FLIERS = False

VALUES = np.array([[i] for i in range(N)])

simple_norms = measure_algo(gossip_algorithm.SimpleGossip, N, ITER, VALUES)
shift_norms = measure_algo(gossip_algorithm.ShiftRegister, N, ITER, VALUES)
jacobi_norms = measure_algo(gossip_algorithm.JacobiAlgo, N, ITER, VALUES)
plt.subplot(1, 3, 1)
sns.boxplot(y=simple_norms, color='g', showfliers=FLIERS)
plt.title('Simple gossip')
plt.subplot(1, 3, 2)
sns.boxplot(y=shift_norms, color='b', showfliers=FLIERS)
plt.title('Shift register')
plt.subplot(1, 3, 3)
sns.boxplot(y=jacobi_norms, color='r', showfliers=FLIERS)
plt.title('Jacobi polynomial')
plt.show()
