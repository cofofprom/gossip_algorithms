from enum import Enum
from graphs import createD
import numpy as np
import daal4py as d4p
import random
from utils import qr_algo


class AlgoType(Enum):
    SIMPLE = 0
    SHIFT_REGISTER = 1
    JACOBI = 2


class GossipAlgo:
    QR_PRECISION = 5
    ROUND_PRECISION = 10

    def __init__(self, start_values, iteration_num, graph_matrix):
        self.start_values = start_values
        self.iter_num = iteration_num
        self.graph_A = graph_matrix
        self.graph_D = createD(self.graph_A)
        self.matrix_W = self.calculate_W()
        self.means = np.array([[np.mean(self.start_values)] for _ in self.start_values])
        self.eigenvalues = sorted(qr_algo(self.matrix_W, self.QR_PRECISION))
        self.gamma = 1 - self.eigenvalues[-2]
        self.x_list = [i for i in range(1, iteration_num)]
        self.y_list = []
        self.measure = 0

    def calculate_W(self):
        return np.eye(self.graph_A.shape[0]) + (self.graph_A - self.graph_D) * 1 / np.max(self.graph_D)


class SimpleGossip(GossipAlgo):
    def __init__(self, start_values, iteration_num, graph_matrix):
        super(SimpleGossip, self).__init__(start_values, iteration_num, graph_matrix)
        self.abs_gamma = min(self.gamma, self.eigenvalues[0] + 1)
        self.measure = self.abs_gamma

    def compute(self):
        values = self.start_values
        for t in range(1, self.iter_num):
            values = self.matrix_W.dot(values)
            diff = values - self.means
            norm = np.round(np.linalg.norm(diff), self.ROUND_PRECISION)
            efficiency = np.exp(np.log(norm + 0.0000001) / t)
            self.y_list.append(efficiency)
        return values


class ShiftRegister(GossipAlgo):
    def __init__(self, start_values, iteration_num, graph_matrix):
        super(ShiftRegister, self).__init__(start_values, iteration_num, graph_matrix)
        self.omega = 2 * (1 - np.sqrt(self.gamma * (1 - (self.gamma / 4)))) / ((1 - (self.gamma / 2)) ** 2)
        self.omega_measure = 1 + (1 / self.graph_A.shape[0]) * (
                (4 - np.sqrt(4 - (1 + (self.eigenvalues[-2]) ** 2))) / (1 + self.eigenvalues[-2]) - 2)
        #print(f"DEBUG: omega = {self.omega}, measure = {self.omega_measure}")
        self.measure = self.omega_measure

    def compute(self):
        curr, prev = self.matrix_W.dot(self.start_values), self.start_values
        efficiency = np.linalg.norm(curr - self.means)
        self.y_list.append(efficiency)
        for t in range(2, self.iter_num):
            iter_values = self.omega * self.matrix_W.dot(curr) + (1 - self.omega) * prev
            curr, prev = iter_values, curr
            np.round(np.linalg.norm(curr - self.means), self.ROUND_PRECISION)
            self.y_list.append(np.exp(np.log(efficiency + 0.0000001) / t))
        return curr


class JacobiAlgo(GossipAlgo):
    T_LIM = 1000
    P_REPEAT = 100
    # стабилизация частоты

    def __init__(self, start_values, iteration_num, graph_matrix):
        super(JacobiAlgo, self).__init__(start_values, iteration_num, graph_matrix)
        self.d = self.calculate_spectral()
        print(f"DEBUG: d = {self.d}")
        self.a_0 = (self.d + 4) / (2 * (2 + self.d))
        self.b_0 = self.d / (2 * (2 + self.d))

    def calculate_spectral(self):
        lazy_W = (np.eye(self.matrix_W.shape[0]) + self.matrix_W) * 0.5

        avg_init_vertex_count = 0

        for _ in range(self.P_REPEAT):
            curr_vertex = 0
            for step in range(self.T_LIM):
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

        p_t = avg_init_vertex_count / self.P_REPEAT
        #print(f"DEBUG: p_t = {p_t}")
        #сделать косвенную проверку
        return (-2) * (np.log(p_t / self.T_LIM))
        # return (-2) * (np.log(p_t) / np.log(self.T_LIM))

    def compute(self):
        prev = self.start_values
        curr = self.a_0 * self.matrix_W.dot(self.start_values) + self.b_0 * self.start_values
        efficiency = np.linalg.norm(curr - self.means)
        self.y_list.append(efficiency)
        for t in range(2, self.iter_num):
            a = ((2 * t + (self.d / 2) + 1) * (2 * t + (self.d / 2) + 2)) / (2 * ((t + 1 + (self.d / 2)) ** 2))
            b = ((self.d ** 2) * (2 * t + (self.d / 2) + 1)) / (
                    8 * ((t + 1 + (self.d / 2)) ** 2) * (2 * t + (self.d / 2)))
            c = ((t ** 2) * (2 * t + (self.d / 2) + 2)) / (((t + 1 + (self.d / 2)) ** 2) * (2 * t + (self.d / 2)))

            iter_value = a * self.matrix_W.dot(curr) + b * curr - c * prev

            curr, prev = iter_value, curr

            efficiency = np.round(np.linalg.norm(curr - self.means), self.ROUND_PRECISION)
            #print(f"debug: jacobi efficiency = {efficiency}")
            self.y_list.append(np.exp(np.log(efficiency + 0.0000001) / t))

        return curr
