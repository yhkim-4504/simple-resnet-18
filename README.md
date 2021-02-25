# simple-resnet-18
* torchvision에서 resnet-18 부분만 정리해 구현한 코드입니다.
* 모델 생성 후  진행과정에 따른 shape를 출력합니다.

## 사용방법
``` bash
python model_build.py
```

## 출력결과
``` python
input shape: torch.Size([1, 3, 256, 256]) 
conv1 shape: torch.Size([1, 64, 128, 128])
bn1 shape: torch.Size([1, 64, 128, 128])
relu shape: torch.Size([1, 64, 128, 128])
maxpool shape: torch.Size([1, 64, 64, 64])
layer1 shape: torch.Size([1, 64, 64, 64])
layer2 shape: torch.Size([1, 128, 32, 32])
layer3 shape: torch.Size([1, 256, 16, 16])
layer4 shape: torch.Size([1, 512, 8, 8])
avgpool shape: torch.Size([1, 512, 1, 1])
flatten shape: torch.Size([1, 512])
fc shape: torch.Size([1, 512])
```
