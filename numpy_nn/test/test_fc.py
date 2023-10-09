import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "numpy_nn"))

import unittest

import torch
import numpy as np

from test_layer import test_module_randomly

from numpy_nn.modules.np_nn import FullyConnectedLayer


np_seed = 0
np.random.seed(np_seed)

class TestDataset(unittest.TestCase):

    def setUp(self) -> None:        
        pass

    def test_fc_(self):
        """
        FullyConnectedLayer test
        """
        n_input_neurons = 6
        n_output_neurons = 3
        n_samples = 5

        my_fc_params = torch_fc_params = [n_input_neurons, n_output_neurons]

        test_module_randomly(FullyConnectedLayer(*my_fc_params),
                             torch.nn.Linear(*torch_fc_params),
                             input_shape=[n_samples, n_input_neurons],
                             output_shape=[n_samples, n_output_neurons],
                             random_sampler=np.random.rand)

    

        test_module_randomly(FullyConnectedLayer(*my_fc_params),
                             torch.nn.Linear(*torch_fc_params),
                             input_shape=[n_samples, n_input_neurons],
                             output_shape=[n_samples, n_output_neurons],
                             atol=1e-6,
                             random_sampler=np.random.randn)
        

if __name__ == "__main__":
    unittest.main()