import torch
from torch import Tensor
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from basic_block import conv3x3, conv1x1, BasicBlock

class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer  # batch norm layer

        self.inplanes = 64  # input shape
        self.dilation = 1  # dilation fixed
        self.groups = 1  # groups fixed
        
        # input block
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[BasicBlock], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        
        # downsampling 필요할경우 downsample layer 생성
        if stride != 1 or self.inplanes != planes:  
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.dilation, norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                 dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        print('input shape:', x.shape)
        x = self.conv1(x)
        print('conv1 shape:', x.shape)
        x = self.bn1(x)
        print('bn1 shape:', x.shape)
        x = self.relu(x)
        print('relu shape:', x.shape)
        x = self.maxpool(x)
        print('maxpool shape:', x.shape)

        x = self.layer1(x)
        print('layer1 shape:', x.shape)
        x = self.layer2(x)
        print('layer2 shape:', x.shape)
        x = self.layer3(x)
        print('layer3 shape:', x.shape)
        x = self.layer4(x)
        print('layer4 shape:', x.shape)

        x = self.avgpool(x)
        print('avgpool shape:', x.shape)
        x = torch.flatten(x, 1)
        print('flatten shape:', x.shape)
        x = self.fc(x)
        print('fc shape:', x.shape)

        return x