import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "numpy_nn"))

import unittest
from typing import Callable, List

import torch
import numpy as np

from test_layer import TestLayer
from numpy_nn.modules.np_nn import ReLULayer, SigmoidLayer


np_seed = 0
np.random.seed(np_seed)

class TestActivations(TestLayer):

    def setUp(self) -> None:        
        pass

    def _test_activation(self,
                         my_activation: Callable,
                         torch_activation: Callable,
                         input_dim: List[int],
                         atol: float = 1e-5,
                         random_sampler: Callable = np.random.rand,
                         print_tensors: bool = False,
                         print_results: bool = False):
        """
        Samples input data and output gradient from a uniform
        distribution and tests if the output and input gradients
        are close to the ones computed by pytorch
        """
        self._test_module_randomly(my_activation(), torch_activation(),
                    input_shape=input_dim, output_shape=input_dim,
                    atol=atol, random_sampler=random_sampler,
                    print_tensors=print_tensors, print_results=print_results)

    def _test_relu_with_args(self,
                            input_dim: List[int],
                            atol: float = 1e-5,
                            random_sampler: Callable = np.random.rand,
                            print_tensors: bool = False,
                            print_results: bool = False):
        self._test_activation(ReLULayer, torch.nn.ReLU, input_dim,
                              atol=atol, random_sampler=random_sampler,
                              print_tensors=print_tensors, print_results=print_results)

    def _test_sigmoid_with_args(self,
                                input_dim: List[int],
                                atol: float = 1e-5,
                                random_sampler: Callable = np.random.rand,
                                print_tensors: bool = False,
                                print_results: bool = False):
        self._test_activation(SigmoidLayer, torch.nn.Sigmoid, input_dim,
                              atol=atol, random_sampler=random_sampler,
                              print_tensors=print_tensors, print_results=print_results)

    def test_relu(self):
        """
        ReLU test
        """
        batch_size = 5
        n_channels = 6
        height = 4
        width = 4

        input_dim = (batch_size, n_channels, height, width)
        atol = 1e-10
        n_iters = 3

        for sampler in (np.random.rand, np.random.randn):
            for _ in range(n_iters):
                with self.subTest(sampler = sampler):
                    self._test_relu_with_args(input_dim,
                                              atol,
                                              sampler)

                                                 
    def test_sigmoid(self):
        """
        Sigmoid test
        """
        batch_size = 5
        n_channels = 6
        height = 4
        width = 4

        input_dim = (batch_size, n_channels, height, width)
        atol = 1e-10
        n_iters = 3

        for sampler in (np.random.rand, np.random.randn):
            for _ in range(n_iters):
                with self.subTest(sampler = sampler):
                    self._test_sigmoid_with_args(input_dim,
                                                 atol,
                                                 sampler)
            

if __name__ == "__main__":
    unittest.main()