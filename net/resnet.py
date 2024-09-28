# import torch
import torch.nn as nn
# import torch.nn.functional as F
from .distillblock import Block, CBAM

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101']

def conv_3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv_1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv_1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv_3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv_1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.expans = block.expansion

        # Using Convolution as linear projection to downsample attention_dw1 output to [1, 512, 6, 6]
        # Relu and  MaxPool2d can be remove to be the same as resnet downsample
        self.attention_dw1 = nn.Sequential(
            # CBAM(64*self.expans, 64*self.expans),
            conv_1x1(64 * self.expans, 512 * self.expans, stride=1),
            self._norm_layer(512*self.expans),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=8, stride=8)
        )
        self.attention_dw2 = nn.Sequential( 
                            # CBAM(128*self.expans, 128*self.expans),
                            conv_1x1(128*self.expans,512*self.expans,stride=1),
                            self._norm_layer(512*self.expans),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=4, stride=4)
                            )  
                            
        self.attention_dw3 = nn.Sequential( 
                            # CBAM(256*self.expans, 256*self.expans),
                            conv_1x1(256*self.expans,512*self.expans,stride=1),
                            self._norm_layer(512*self.expans),
                            nn.ReLU(inplace=True),
                            nn.MaxPool2d(kernel_size=2, stride=2) 
                            )
        # last conv to down to num_classes
#        self.last_conv = conv3x3(512*self.expans, num_classes)
        self.last_conv = nn.Linear(512*self.expans, num_classes)
        # global avg-pooling
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))

        # Remove this for running distillation block
        self.block1_fgw = nn.Sequential(
                                        CBAM(64*self.expans,64*self.expans),
                                        Block(64*self.expans, 128*self.expans),
                                        Block(128*self.expans, 256*self.expans),
                                        Block(256*self.expans, 512*self.expans, keep_dim=False))
        self.block2_fgw = nn.Sequential(
                                        CBAM(128*self.expans,128*self.expans),
                                        Block(128*self.expans, 256*self.expans),
                                        Block(256*self.expans, 512*self.expans, keep_dim=False))
        self.block3_fgw = nn.Sequential(
                                        CBAM(256*self.expans,256*self.expans),
                                        Block(256*self.expans, 512*self.expans, keep_dim=False))
        
        # Classifier for each output
#         self.fc1 = nn.Linear(64*self.expans,num_classes)
#         self.fc2 = nn.Linear(128*self.expans,num_classes)
#         self.fc3 = nn.Linear(256*self.expans,num_classes)
#         self.fc4 = nn.Linear(512*self.expans, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv_1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        out1 = x
        att1 = self.attention_dw1(out1)
        out1 = self.block1_fgw(out1)
        out1 = out1*att1
        out1 = self.avgp(out1)
        out1 = out1.view((out1.shape[0], -1))
        fea1 = out1
        out1 = self.last_conv(out1)

        x = self.layer2(x)
        out2 = x
        att2 = self.attention_dw2(out2)
#         print(att2.shape)
        out2 = self.block2_fgw(out2)
#         print(out2.shape)
        out2 = out2*att2
        out2 = self.avgp(out2)
        out2 = out2.view((out2.shape[0], -1))
        fea2 = out2
        out2 = self.last_conv(out2)

        x = self.layer3(x)
        out3 = x
        att3 = self.attention_dw3(out3)
#         print(att3.shape)
        out3 = self.block3_fgw(out3)
#         print(out3.shape)
        out3 = out3*att3
        out3 = self.avgp(out3)
        out3 = out3.view((out3.shape[0], -1))
        fea3 = out3
        out3 = self.last_conv(out3)

        x = self.layer4(x)
        out4 = x
        out4 = self.avgp(out4)
        out4 = out4.view((out4.shape[0], -1))
        fea4 = out4
        out4 = self.last_conv(out4)

        return [out4, out3, out2, out1], [fea4, fea3, fea2, fea1]

def _resnet(arch, block, layers, pretrained, progress, num_classes, **kwargs):
    model = ResNet(block, layers, num_classes=num_classes, **kwargs)

    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


if __name__ == '__main__':
    model = resnet18(num_classes=10)
    print(model)