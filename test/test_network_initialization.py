from unittest import TestCase

import network
from network import network
from network.network import initialize_coefficients
import numpy as np
import os

working_directory = os.path.dirname(__file__)
    
class TestInitialization(TestCase):
    
    @classmethod
    def setUpClass(self):
        np.random.seed(3)
        self.weights, self.biases = initialize_coefficients([5,4,3])
                
    def test_weight_initializations(self):
        l1_weights = np.load(os.path.join(working_directory, "resources/initial_weights_l1.npy"))
        l2_weights = np.load(os.path.join(working_directory, "resources/initial_weights_l2.npy"))
      
        self.assertTrue(np.array_equal(l1_weights, self.weights[0]))
        self.assertTrue(np.array_equal(l2_weights, self.weights[1]))

    def test_bias_initializations(self):
        l1_biases = np.load(os.path.join(working_directory, "resources/initial_biases_l1.npy"))
        l2_biases = np.load(os.path.join(working_directory, "resources/initial_biases_l2.npy"))

        self.assertTrue(np.array_equal(l1_biases, self.biases[0]))
        self.assertTrue(np.array_equal(l2_biases, self.biases[1]))
        
