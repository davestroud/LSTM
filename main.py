#!/usr/bin/python3
# https://github.com/Manik9/LSTMs/blob/master/lstm.py

import random as np
import math

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values):
    return values*(1-values)

def tanh_derivatives(values):
    return 1. - values ** 2

# create uniform random array w/ values in [a,b] and shape args

def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct
        # weight matrices
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
