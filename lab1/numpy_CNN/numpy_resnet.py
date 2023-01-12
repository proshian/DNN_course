from typing import List

import numpy as np
from NumpyNN.NN_np import (
    # Conv2dWithLoops as Conv2d,
    Conv2d,
    ReLULayer,
    Sequential,
    MaxPool2d,
    FullyConnectedLayer,
    TrainableLayer,
    Flatten,
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
        
        self.trainable_layers = [self.conv1, self.conv2, self.conv3]
        if self.conv_to_match_dimensions is not None:
            self.trainable_layers.append(self.conv_to_match_dimensions)
        

        self.trainable_layers = [self.conv1, self.conv2, self.conv3]
        

    def forward(self, input_: np.ndarray) -> np.ndarray:
        self.input_ = input_
        
        if self.conv_to_match_dimensions is not None:
            identity = self.conv_to_match_dimensions.forward(input_)
        else:
            identity = input_

        # The layers from conv1 to conv3 are main path.
        #! Maybe make them members of a sequential network object?
        out = self.conv1.forward(input_)
        out = self.relu1.forward(out)

        out = self.conv2.forward(out)
        out = self.relu2.forward(out)

        out = self.conv3.forward(out)

        out += identity
        out = self.relu3.forward(out)

        return out
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        main_path_output_gradient = self.relu3.backward(output_gradient)
        identity_output_gradient = main_path_output_gradient.copy()
        
        # The layers from conv1 to conv3 are main path.
        #! Maybe make them members of a sequential network object?
        for layer in ([self.conv3, self.relu2, self.conv2, self.relu1, self.conv1]):
            main_path_output_gradient = layer.backward(main_path_output_gradient)
        
        if self.conv_to_match_dimensions is not None:
            identity_output_gradient = self.conv_to_match_dimensions.backward(identity_output_gradient)
        
        return main_path_output_gradient + identity_output_gradient

    def get_trainable_layers(self) -> List[TrainableLayer]:
        return self.trainable_layers
    

class ResNet:
    """
    ResNet model.

    Attributes:
        block: Building block type. Currently Bottleneck only.
        block_nums: Number of blocks for each block configuration.
            For example for ResNet-50, n_blocks = [3, 4, 6, 3].
        n_classes (int): Number of classes.
        img_channels (int): Number of channels in the input image.
    """

    def __init__(
        self,
        # ! In the future Basic Residual Block may be included as a possible type
        block: Bottleneck,
        block_nums: List[int],
        n_classes: int,
        img_channels: int = 3,
    ) -> None:
        self.cur_block_in_channels = 64
        self.conv1 = Conv2d(
            img_channels, self.cur_block_in_channels,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = ReLULayer()
        self.conv2_x = self._make_blocks(block_nums[0], 64, False)
        self.conv3_x = self._make_blocks(block_nums[1], 128)
        self.conv4_x = self._make_blocks(block_nums[2], 256)
        self.conv5_x = self._make_blocks(block_nums[3], 512)
        # ! add adaptive avg pool
        # ! Maybe add reshape
        self.fc = FullyConnectedLayer(512 * block.expansion, n_classes)


        nn_modules = [
            self.conv1, self.conv2_x, self.conv3_x,
            self.conv4_x, self.conv5_x, self.fc
        ]

        self.trainable_layers = []
        for module in nn_modules:
            self.trainable_layers.extend(module.get_trainable_layers())
        
    def get_trainable_layers(self) -> List[TrainableLayer]:
        return self.trainable_layers


    def _make_blocks(
        self,
        n_blocks: int,
        bottleneck_depth: int,
        downsampling: bool = True
    ):
        """
        Creates a sequence of blocks for a specific stage of ResNet.
        Args:
            n_blocks (int): Number of blocks in the stage.
            first_block_in_channels (int): Number of input channels for the first block.
            downsampling (bool): Whether to downsample the feature map.
                If True, the feature map is downsampled by a factor of 2
                in the first convolution of the first block. downsampling is
                supposed to be True for conv3_x, conv4_x, conv5_x, and False
                for conv2_x.
        Returns:
            nn.Sequential: A sequence of blocks. For example
                all blocks of conv_1_x.
        """
        
        blocks = []
        stride_for_downsampling = 2 if downsampling else 1
        block = Bottleneck(self.cur_block_in_channels, bottleneck_depth, stride_for_downsampling)
        self.cur_block_in_channels = bottleneck_depth * block.expansion
        blocks.append(block)
        for _ in range(1, n_blocks):
            block = Bottleneck(self.cur_block_in_channels, bottleneck_depth)
            blocks.append(block)
        return Sequential(blocks)
    
    def forward(self, input_: np.ndarray) -> np.ndarray:
        out = self.conv1.forward(input_)
        out = self.maxpool.forward(out)
        out = self.relu.forward(out)
        out = self.conv2_x.forward(out)
        out = self.conv3_x.forward(out)
        out = self.conv4_x.forward(out)
        out = self.conv5_x.forward(out)
        out = out.reshape(out.shape[0], -1) # ! temporary solution 
        out = self.fc.forward(out)
        return out
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        out = self.fc.backward(output_gradient)
        # ! Temporary solution. 2048 = number of channels after
        # last block of conv5_3 = 512 * block.expansion
        out = out.reshape(out.shape[0], 2048, 1, 1)
        out = self.conv5_x.backward(out)
        out = self.conv4_x.backward(out)
        out = self.conv3_x.backward(out)
        out = self.conv2_x.backward(out)
        out = self.relu.backward(out)
        out = self.maxpool.backward(out)
        out = self.conv1.backward(out)
        return out

def resnet101(n_classes: int, img_channels: int = 3) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], n_classes, img_channels)