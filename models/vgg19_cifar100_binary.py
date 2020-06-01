import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d



class VGG19_Cifar100(nn.Module):

    def __init__(self, num_classes=100):
        super(VGG19_Cifar100, self).__init__()
        self.features = nn.Sequential(

            # Block 1
            BinarizeConv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            BinarizeConv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Block 3
            BinarizeConv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Block 4
            BinarizeConv2d(256, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Block 5
            BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(512, num_classes, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'lr': 5e-3},
            50: {'lr': 1e-3},
            100: {'lr': 5e-4},
            150: {'lr': 1e-4},
            200: {'lr': 5e-5},
            250: {'lr': 1e-5},
            300: {'lr': 5e-6},
            350: {'lr': 1e-6}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x


def vgg19_cifar100_binary(**kwargs):
    num_classes = kwargs.get( 'num_classes', 100)
    return VGG19_Cifar100(num_classes)
