import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "numpy_nn"))

import unittest
from typing import Callable

import torch
import numpy as np

from test_layer import TestLayer
from numpy_nn.modules.np_nn import MaxPool2d


# np_seed = 0
# torch_seed = 0
# np.random.seed(np_seed)
# torch.manual_seed(torch_seed)

class TestMaxPool2d(TestLayer):

    def setUp(self) -> None:        
        pass

    def _test_maxpool_with_args(self, batch_size: int, height: int,
                               width: int, n_channels: int, kernel_size: int,
                               stride: int, padding: int, atol: float = 1e-5,
                               random_sampler: Callable = np.random.rand,
                               print_tensors: bool = False,
                               print_results: bool = False):
        output_width = (width + 2 * padding - kernel_size) // stride + 1
        output_height = (height + 2 * padding - kernel_size) // stride + 1

        my_pool_args = torch_pool_args = [kernel_size, stride, padding]

        my_pool = MaxPool2d(*my_pool_args)
        torch_pool = torch.nn.MaxPool2d(*torch_pool_args)

        input_shape = [batch_size, n_channels, height, width]
        output_shape=[batch_size, n_channels, output_height, output_width]


        self._test_module_randomly(
            my_pool,
            torch_pool,
            input_shape=input_shape,
            output_shape=output_shape,
            atol=atol,
            random_sampler = random_sampler,
            print_tensors=print_tensors,
            print_results=print_results)


    def test_maxpool_1(self):
        """
        Maxpool test
        """
        batch_size = 10
        n_channels = 3
        height = 16
        width = 16

        kernel_size = 3
        stride = 2
        padding = 1

        n_iters = 3

        for sampler in (np.random.rand, np.random.randn):
            for _ in range(n_iters):
                with self.subTest(sampler = sampler):
                    self._test_maxpool_with_args(batch_size,
                                                height,
                                                width,
                                                n_channels,
                                                kernel_size,
                                                stride,
                                                padding,
                                                atol=1e-6,
                                                random_sampler=sampler)


    def test_maxpool_2(self):
        """
        Maxpool test
        """
        batch_size = 2
        n_channels = 6
        height = 4
        width = 3

        kernel_size = 2
        stride = 1
        padding = 0

        n_iters = 3

        for sampler in (np.random.rand, np.random.randn):
            for _ in range(n_iters):
                with self.subTest(sampler = sampler):
                    self._test_maxpool_with_args(batch_size,
                                                height,
                                                width,
                                                n_channels,
                                                kernel_size,
                                                stride,
                                                padding,
                                                atol=1e-6,
                                                random_sampler=sampler)
            

if __name__ == "__main__":
    unittest.main()