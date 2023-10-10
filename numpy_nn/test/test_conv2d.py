import sys; import os; sys.path.insert(1, os.path.join(os.getcwd(), "numpy_nn"))

import unittest

import torch
import numpy as np

from test_layer import TestLayer

from numpy_nn.modules.np_nn import Conv2d, Conv2dWithLoops


np_seed = 0
np.random.seed(np_seed)

class TestFullyConnectedLayer(TestLayer):

    def setUp(self) -> None:        
        pass

    def test_conv2d_efficient(self):
        self._test_conv2d(Conv2d)
    
    def test_conv2d_with_loops(self):
        self._test_conv2d(Conv2dWithLoops)

    def _test_conv2d(self, my_conv2d_constructor):
        """
        Conv2d test
        """
        batch_size = 5
        n_input_channels = 4
        n_output_channels = 2
        input_width = 3
        input_height = 5

        kernel_size = 3
        stride = 1
        padding = 1

        n_iters = 3


        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1

        input_shape = (batch_size, n_input_channels, input_height, input_width)
        output_shape = (batch_size, n_output_channels, output_height, output_width)

        for sampler in (np.random.rand, np.random.randn):
            for _ in range(n_iters):
                input_np = sampler(*input_shape).astype(np.float32)
                dJ_dout = sampler(*output_shape)

                for bias in (True, False):

                    my_conv2d_kwargs = torch_conv2d_kwargs = {
                        "in_channels": n_input_channels,
                        "out_channels": n_output_channels,
                        "kernel_size": kernel_size,
                        "stride": stride,
                        "padding": padding,
                        "bias": bias
                    }

                    my_conv2d = my_conv2d_constructor(**my_conv2d_kwargs)
                    torch_conv2d = torch.nn.Conv2d(**torch_conv2d_kwargs)

                    with self.subTest(input_np = input_np,
                                      dJ_dout = dJ_dout,
                                      sampler = sampler,
                                      bias = bias):
                        self._test_module(
                            my_conv2d,
                            torch_conv2d,
                            input_np = input_np,
                            dJ_dout = dJ_dout,
                            atol=1e-6)
            

if __name__ == "__main__":
    unittest.main()