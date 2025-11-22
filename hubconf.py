"""PaddleHub 模型配置"""

import paddle.nn as nn


def resnet(pretrained=False, **kwargs):
    """简单的 ResNet 模型"""
    return nn.Sequential(
        nn.Conv2D(3, 64, 7, stride=2, padding=3),
        nn.BatchNorm2D(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2D((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 1000)
    )


def mobilenet(pretrained=False, **kwargs):
    """简单的 MobileNet 模型"""
    return nn.Sequential(
        nn.Conv2D(3, 32, 3, stride=2, padding=1),
        nn.BatchNorm2D(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2D((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 1000)
    )
