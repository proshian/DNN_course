import numpy as np
from NumpyNN.NN_np import (
    Conv2d,
    ReLULayer,
)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> Conv2d:
    return Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> Conv2d:
    """3x3 "same" convolution"""
    return Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

# Bottleneck numpy only
class Bottleneck:
    """
    A "bottleneck" building block of a ResNet.
    This block always consists of a sequence of three convolutions:
    1x1, 3x3, 1x1. For example if out_channels = 64, then the sequence is
    [
        1x1, 64
        3x3, 64
        1x1, 256
    ]

    Attributes:
        in_channels: The number of input channels of the first convolution.
        bottleneck_depth: The number of output channels (for the first two convolutions).
        stride_for_downsampling: The stride for downsampling the input.
    """

    # The number of output channels for the last convolution is always 4 times
    # more than for than for other convolutions. 
    expansion: int = 4

    def __init__(self, in_channels: int, bottleneck_depth: int, stride_for_downsampling: int = 1):
        self.in_channels = in_channels
        self.bottleneck_depth = bottleneck_depth
        self.stride_for_downsampling = stride_for_downsampling

        self.conv1 = conv1x1(in_channels, bottleneck_depth, stride_for_downsampling)
        self.conv2 = conv3x3(bottleneck_depth, bottleneck_depth)
        self.conv3 = conv1x1(bottleneck_depth, bottleneck_depth * self.expansion)
        self.relu1 = ReLULayer()
        self.relu2 = ReLULayer()
        self.relu3 = ReLULayer()

        # conv_to_match_dimensions is created only if it's needed to not waste memory.
        # There are two cases for performing a convolution on identity:
        # 1. There will be downsampling (stride_for_downsampling != 1)
        # 2. The number of bottleneck's output channels is different from the
        #    number of input channels (in_channels != bottleneck_depth * self.expansion)
        # Note that the number of input's channels is equal to the number
        # of output's channels when it's not the first bottleneck in a block.
        self.conv_to_match_dimensions = None
        if in_channels != bottleneck_depth * self.expansion or stride_for_downsampling != 1:
            self.conv_to_match_dimensions = conv1x1(in_channels, bottleneck_depth * self.expansion, stride_for_downsampling)
        

    def forward(self, input_: np.ndarray) -> np.ndarray:
        """
        Performs a forward pass of the bottleneck.
        """
        
        if self.conv_to_match_dimensions:
            identity = self.conv_to_match_dimensions.forward(input_)
        else:
            identity = input_  # ! thought about using np.copy(input_) here but convolutions are not inplace

        out = self.conv1.forward(input_)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)

        out = self.conv3.forward(out)

        out += identity
        out = self.relu3.forward(out)

        return out
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Performs a backward pass of the bottleneck.
        """
        # the last operation in forward pass is relu
        # so the first operation in backward pass is relu_backward
        addition_output_gradient = self.relu.backward(output_gradient)
        

"""
class Add(Layer):
    def __init__(self):
        self.left = None
        self.right = None

    def forward(self, x: np.array):
        assert len(x) == 2
        self.left = x[0]
        self.right = x[1]

        return self.left + self.right

    def backward(self, d_output: np.array):
        return np.array([d_output, d_output])


class ResidualBlock(Layer):
    def __init__(self, layers, downsample):
        super().__init__('ResidualBlock')
        self.layers = layers
        self.add_layer = Add()
        self.downsample = downsample

    def forward(self, x: np.array):
        self.input = x

        fwd = self.layers(x)
        return self.add_layer([fwd, self.downsample(self.input)])

    def backward(self, d_output: np.array):
        dl, dr = self.add_layer.backward(d_output)
        d_path = self.layers.backward(dl)
        d_downsample = self.downsample.backward(dr)

        return d_path + d_downsample

    def params(self):
        return self.layers.params() + self.downsample.params()
"""