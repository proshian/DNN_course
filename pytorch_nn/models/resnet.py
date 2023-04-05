from typing import List

import torch  # Used only for flattening
import torch.nn as nn
from torch import Tensor  # For typing

# torch.use_deterministic_algorithms(True)


__all__ = ['ResNet', 'resnet101', 'Bottleneck']  # ! Bottleneck is temporary here

def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding (to reserve feature map size)"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)

def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
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

    def __init__(
            self, in_channels: int, bottleneck_depth: int,
            stride_for_downsampling: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.bottleneck_depth = bottleneck_depth
        self.stride_for_downsampling = stride_for_downsampling

        self.bn1 = nn.BatchNorm2d(bottleneck_depth)
        self.bn2 = nn.BatchNorm2d(bottleneck_depth)
        self.bn3 = nn.BatchNorm2d(bottleneck_depth * self.expansion)

        self.conv1 = conv1x1(in_channels, bottleneck_depth, stride_for_downsampling)
        self.conv2 = conv3x3(bottleneck_depth, bottleneck_depth)
        self.conv3 = conv1x1(bottleneck_depth, bottleneck_depth * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # conv_to_match_dimensions is created only if it's needed to not waste memory.
        # ! However it seems like it doesn't affect the memory usage in pytorch implementation.
        # There are two cases for performing a convolution on identity:
        # 1. There will be downsampling (stride_for_downsampling != 1)
        # 2. The number of bottleneck's output channels is different from the
        #    number of input channels (in_channels != bottleneck_depth * self.expansion)
        # Note that the number of input's channels is equal to the number
        # of output's channels when it's not the first bottleneck in a block. 
        self.conv_to_match_dimensions = None
        if in_channels != bottleneck_depth * self.expansion or stride_for_downsampling != 1:
            self.conv_to_match_dimensions = conv1x1(in_channels, bottleneck_depth * self.expansion, stride_for_downsampling)
            self.bn_for_residual = nn.BatchNorm2d(bottleneck_depth * self.expansion)
        # ! Возможно, стоит вынести проверку на необходимость применения conv_to_match_dimensions
        # отсюда в ResNet и передавать в Bottleneck параметр need_to_match_dimensions
    
    def forward(self, x: Tensor) -> Tensor:

        if self.conv_to_match_dimensions is not None:
            identity = self.conv_to_match_dimensions(x)
            identity = self.bn_for_residual(identity)
        else:
            identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """
    ResNet model.

    Attributes:
        block (nn.Module): Building block type. Currently Bottleneck only.
        block_nums: Number of blocks for each block configuration.
            For example for ResNet-50, n_blocks = [3, 4, 6, 3].
        n_classes (int): Number of classes.
        img_channels (int): Number of channels in the input image.

    """

    def __init__(
        self,
        block: nn.Module,
        block_nums: List[int],
        n_classes: int,
        img_channels: int = 3,
    ) -> None:
        super().__init__()
        self.cur_block_in_channels = 64
        self.conv1 = nn.Conv2d(
            img_channels, self.cur_block_in_channels,
            kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.cur_block_in_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_x = self._make_blocks(block_nums[0], 64, False)
        self.conv3_x = self._make_blocks(block_nums[1], 128)
        self.conv4_x = self._make_blocks(block_nums[2], 256)
        self.conv5_x = self._make_blocks(block_nums[3], 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, n_classes)

    def _make_blocks(
        self,
        n_blocks: int,
        bottleneck_depth: int,
        downsampling: bool = True
    ) -> nn.Sequential:
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
        for i in range(1, n_blocks):
            block = Bottleneck(self.cur_block_in_channels, bottleneck_depth)
            blocks.append(block)
        return nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet101(n_classes: int, img_channels: int) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], n_classes, img_channels)