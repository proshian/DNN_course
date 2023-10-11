import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "numpy_nn"))

import unittest
from typing import Callable, List

import torch
import numpy as np

from test_layer import TestLayer
from numpy_nn.modules.np_nn import Flatten


# np_seed = 0
# torch_seed = 0
# np.random.seed(np_seed)
# torch.manual_seed(torch_seed)

class TestMaxPool2d(TestLayer):

    def setUp(self) -> None:        
        pass

    def _test_flatten_with_args(self,
                                input_shape: List[int],
                                atol: float = 1e-6,
                                random_sampler: Callable = np.random.rand,
                                print_tensors: bool = False,
                                print_results: bool = False):
    
        batch_size, *rest_input_dim = input_shape
        output_shape = [batch_size, np.prod(rest_input_dim)]

        self._test_module_randomly(Flatten(),
                                   torch.nn.Flatten(),
                                   input_shape=input_shape,
                                   output_shape=output_shape,
                                   atol=atol,
                                   random_sampler=random_sampler,
                                   print_tensors=print_tensors,
                                   print_results=print_results)

    def test_flatten(self):
        """
        Maxpool test
        """
        input_shape = [2, 3, 5, 5]

        n_iters = 3

        for sampler in (np.random.rand, np.random.randn):
            for _ in range(n_iters):
                with self.subTest(sampler = sampler):
                    self._test_flatten_with_args(input_shape,
                                                atol=1e-6,
                                                random_sampler=sampler)
            

if __name__ == "__main__":
    unittest.main()