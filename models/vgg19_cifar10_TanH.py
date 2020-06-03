import torch.nn as nn
import torchvision.transforms as transforms

class VGG19_Cifar10_TanH(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG19_Cifar10_TanH, self).__init__()
        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.Hardtanh(inplace=True),            
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes, bias=True),
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


def vgg19_cifar10_TanH(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    return VGG19_Cifar10_TanH(num_classes)
