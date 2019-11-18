 #!/usr/bin/python3


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


