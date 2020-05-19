import numpy as np

def initialize_coefficients(layers):
    L = len(layers)
    weights = [np.random.randn(layers[l], layers[l - 1]) * 0.01 for l in range(1, L)]
    biases = [np.zeros((layers[l], 1)) for l in range(1, L)]
    return weights, biases
