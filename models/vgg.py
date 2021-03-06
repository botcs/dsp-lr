# Base code: PyTorch models in torchvision
# Original source: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math


__all__ = [
    'VGG',
    'vgg11', 'vgg11_bn',
    'vgg13', 'vgg13_bn',
    'vgg16', 'vgg16_bn', 'dil_vgg16_bn', 'dil_vgg16_bn_selu',
    'vgg19_bn', 'vgg19', 'dil_vgg19_bn', 'dil_vgg19_bn_selu', 'dil_vgg19_bn_selu_halved',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=3, init_weights=True, num_features=512):
        super(VGG, self).__init__()
        self.features = features
        self.num_features = num_features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(self.num_features, num_classes, 7),
            nn.AdaptiveAvgPool1d(1)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, in_channels=1, dilated=False, selu=False):
    layers = []
    dilated_factor=1
    def act_fn():
        return nn.SELU(inplace=True) if selu else nn.ReLU(inplace=True)
    def drop_fn(p=0.2):
        return nn.AlphaDropout(p) if selu else nn.Dropout(p)
    for v in cfg:
        if v == 'M':
            dilated_factor = 1
            layers += [nn.MaxPool1d(kernel_size=2, stride=2)]
        else:
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1,
                               dilation=dilated_factor)
            #dropout = drop_fn()
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), act_fn()]
            else:
                layers += [conv1d, act_fn()]
            in_channels = v
            if dilated:
                dilated_factor *= 2
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M', 256, 256, 256, 256, 'M'],
}


def vgg11(in_channels=1, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(in_channels=1, pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True, in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(in_channels=1, pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(in_channels=1, pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True, in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(in_channels=1, pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(in_channels=1, pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True, in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model

def dil_vgg16_bn(in_channels=1, pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True, in_channels=in_channels, dilated=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dil_vgg16_bn']))
    return model

def dil_vgg16_bn_selu(in_channels=1, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True, in_channels=in_channels,
                            dilated=True, selu=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dil_vgg19_bn_selu']))
    return model


def vgg19(in_channels=1, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(in_channels=1, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True, in_channels=in_channels), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model

def dil_vgg19_bn(in_channels=1, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True, in_channels=in_channels, dilated=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dil_vgg19_bn']))
    return model

def dil_vgg19_bn_selu(in_channels=1, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True, in_channels=in_channels,
                            dilated=True, selu=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dil_vgg19_bn_selu']))
    return model

def dil_vgg19_bn_selu_halved(in_channels=1, pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['F'], batch_norm=True, in_channels=in_channels,
                            dilated=True, selu=True), num_features=256, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dil_vgg19_bn_selu']))
    return model
