import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "numpy_nn"))

import unittest

import torch
import numpy as np

from test_layer import TestLayer
from numpy_nn.modules.np_nn import FullyConnectedLayer


# np_seed = 0
# torch_seed = 0
# np.random.seed(np_seed)
# torch.manual_seed(torch_seed)

class TestFullyConnectedLayer(TestLayer):

    def setUp(self) -> None:        
        pass

    def test_fc(self):
        """
        FullyConnectedLayer test
        """
        n_input_neurons = 6
        n_output_neurons = 3
        n_samples = 5
        n_iters = 3

        my_fc_params = torch_fc_params = (n_input_neurons, n_output_neurons)
        input_shape = (n_samples, n_input_neurons)
        output_shape = (n_samples, n_output_neurons)

        for sampler in (np.random.rand, np.random.randn):
            for _ in range(n_iters):
                input_np = sampler(*input_shape).astype(np.float32)
                dJ_dout = sampler(*output_shape)

                with self.subTest(input_np = input_np, dJ_dout = dJ_dout, sampler = sampler):
                    self._test_module(
                        my_module = FullyConnectedLayer(*my_fc_params),
                        torch_module = torch.nn.Linear(*torch_fc_params),
                        input_np = input_np,
                        dJ_dout = dJ_dout,
                        atol=1e-6)
        

if __name__ == "__main__":
    unittest.main()