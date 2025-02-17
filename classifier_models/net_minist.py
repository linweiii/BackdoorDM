import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn import Module
from torchvision import transforms

class Conv2dBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(Conv2dBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, in_c, out_c, ker_size=(3, 3), stride=1, padding=1, batch_norm=True, relu=True):
        super(ConvTranspose2dBlock, self).__init__()
        self.convtranpose2d = nn.ConvTranspose2d(in_c, out_c, ker_size, stride, padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if relu:
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class DownSampleBlock(nn.Module):
    def __init__(self, ker_size=(2, 2), stride=2, dilation=(1, 1), ceil_mode=False, p=0.0):
        super(DownSampleBlock, self).__init__()
        self.maxpooling = nn.MaxPool2d(kernel_size=ker_size, stride=stride, dilation=dilation, ceil_mode=ceil_mode)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x


class UpSampleBlock(nn.Module):
    def __init__(
        self, in_c, out_c, kernel_size, stride, padding, scale_factor=(2, 2), mode="bilinear", batch_norm=True, p=0.0
    ):
        super(UpSampleBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)
        self.conv2d = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_c, eps=1e-5, momentum=0.05, affine=True)
        if p:
            self.dropout = nn.Dropout(p)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x

class Normalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = (x[:, channel] - self.expected_values[channel]) / self.variance[channel]
        return x_clone


class Denormalize:
    def __init__(self, opt, expected_values, variance):
        self.n_channels = opt.input_channel
        self.expected_values = expected_values
        self.variance = variance
        assert self.n_channels == len(self.expected_values)

    def __call__(self, x):
        x_clone = x.clone()
        for channel in range(self.n_channels):
            x_clone[:, channel] = x[:, channel] * self.variance[channel] + self.expected_values[channel]
        return x_clone


class Normalizer:
    def __init__(self, opt):
        self.normalizer = self._get_normalizer(opt)

    def _get_normalizer(self, opt):
        if opt.dataset == "cifar10":
            normalizer = Normalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            normalizer = Normalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            normalizer = None
        else:
            raise Exception("Invalid dataset")
        return normalizer

    def __call__(self, x):
        if self.normalizer:
            x = self.normalizer(x)
        return x


class Denormalizer:
    def __init__(self, opt):
        self.denormalizer = self._get_denormalizer(opt)

    def _get_denormalizer(self, opt):
        if opt.dataset == "cifar10":
            denormalizer = Denormalize(opt, [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif opt.dataset == "mnist":
            denormalizer = Denormalize(opt, [0.5], [0.5])
        elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
            denormalizer = None
        else:
            raise Exception("Invalid dataset")
        return denormalizer

    def __call__(self, x):
        if self.denormalizer:
            x = self.denormalizer(x)
        return x


# ---------------------------- Classifiers ----------------------------#
class MNISTBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(MNISTBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ind = None

    def forward(self, x):
        return self.conv1(F.relu(self.bn1(x)))


class NetC_MNIST(nn.Module):
    def __init__(self):
        super(NetC_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3), 2, 1)  # 14
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = MNISTBlock(32, 64, 2)  # 7
        self.layer3 = MNISTBlock(64, 64, 2)  # 4
        self.flatten = nn.Flatten()
        self.linear6 = nn.Linear(64 * 4 * 4, 512)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout8 = nn.Dropout(0.3)
        self.linear9 = nn.Linear(512, 10)

    def forward(self, x):
        for module in self.children():
            x = module(x)
        return x