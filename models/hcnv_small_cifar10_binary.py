import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d
import os



class hcnv_small_cifar10(nn.Module):

    def __init__(self, num_classes=10):
        super(hcnv_small_cifar10, self).__init__()
        self.features = nn.Sequential(

            # Block 1
            BinarizeConv2d(3, 16, kernel_size=5, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            BinarizeConv2d(16, 32, kernel_size=5, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.Hardtanh(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Block 3
            BinarizeConv2d(32, 64, kernel_size=5, padding=0, bias=True),
            nn.BatchNorm2d(64),
            nn.Hardtanh(inplace=True),
        )
        self.classifier = nn.Sequential(
            BinarizeLinear(64, 128, bias=True),
            nn.BatchNorm1d(128, affine=False),

            BinarizeLinear(128, 128, bias=True),
            nn.BatchNorm1d(num_classes, affine=False),

            BinarizeLinear(128, num_classes, bias=False),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-2},
            50: {'lr': 1e-2},
            100: {'lr': 5e-3},
            150: {'lr': 1e-3},
            200: {'lr': 5e-4},
            250: {'lr': 1e-4},
            300: {'lr': 5e-5},
            350: {'lr': 1e-5},
            400: {'lr': 5e-6},
            450: {'lr': 1e-6}
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 64)
        x = self.classifier(x)
        return x

    def export(self,path):
        import numpy as np
        dic = {}
        i = 0
        
        # process conv and BN layers
        for k in range(len(self.features)):
            if hasattr(self.features[k], 'weight') and not hasattr(self.features[k], 'running_mean'):
                dic['arr_'+str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
            elif hasattr(self.features[k], 'running_mean'):
                dic['arr_'+str(i)] = self.features[k].bias.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.features[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = 1./np.sqrt(self.features[k].running_var.detach().numpy())
                i = i + 1
        
        # process linear and BN layers
        for k in range(len(self.classifier)):
            if hasattr(self.classifier[k], 'weight') and not hasattr(self.classifier[k], 'running_mean'):
                dic['arr_'+str(i)] = np.transpose(self.classifier[k].weight.detach().numpy())
                i = i + 1
                if(self.classifier[k].bias != None):
                  dic['arr_'+str(i)] = self.classifier[k].bias.detach().numpy()
                  i = i + 1
            elif hasattr(self.classifier[k], 'running_mean'):
                dic['arr_'+str(i)] = self.classifier[k].bias.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].weight.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = self.classifier[k].running_mean.detach().numpy()
                i = i + 1
                dic['arr_'+str(i)] = 1./np.sqrt(self.classifier[k].running_var.detach().numpy())
                i = i + 1
        
        save_file = os.path.join(path, 'model_best.npz')
        np.savez(save_file, **dic)
        print("Model exported at: ", save_file)


def hcnv_small_cifar10_binary(**kwargs):
    num_classes = kwargs.get( 'num_classes', 10)
    return hcnv_small_cifar10(num_classes)
