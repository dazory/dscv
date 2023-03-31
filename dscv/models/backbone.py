import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

'''
name:
    AlexNet         : ['alexnet']
    VGG             : ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
    ResNet          : ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    SqueezeNet      : ['squeezenet1_0', 'squeezenet1_1']
    DenseNet        : ['densenet121', 'densenet161', 'densenet169', 'densenet201']
    Inception v3    : ['inception_v3']
    GoogLeNet       : ['googlenet']
    ShuffleNet v2   : ['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
    MobileNet v2    : ['mobilenet_v2']
    ResNeXt         : ['resnext50_32x4d', 'resnext101_32x8d']
    Wide ResNet     : ['wide_resnet50_2', 'wide_resnet101_2']
    MNASNet         : ['mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3']
'''


def build_backbone(name, pretrained=True, progress=True, return_layers=None, **kwargs):
    '''
    return_layers : e.g. {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
    '''
    backbone = getattr(torchvision.models, name)(pretrained=pretrained, progress=progress, **kwargs)
    if return_layers is not None:
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    return backbone

