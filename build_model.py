from resnet_18 import ResNet
from basic_block import BasicBlock
import torch

if __name__ == '__main__':
    model = ResNet(BasicBlock, layers=[2, 2, 2, 2], num_classes=512)
    x = torch.randn(1, 3, 256, 256)
    print('\noutput shape:', model(x).shape)