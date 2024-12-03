import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torch.nn.init as init

class NET_1k_ECG(nn.Module):
    def __init__(self, num_classes):
        super(NET_1k_ECG, self).__init__()

        cfg_cnn = [(1, 5, 1, 4, 7, 2),(5, 5, 1, 1, 5, 1),(5, 5, 1, 1, 3, 1)]
        
        self.quant = torch.quantization.QuantStub() 
        self.dequant = torch.quantization.DeQuantStub()

        # Define layers based on configuration
        self.conv1_depthwise = nn.Conv1d(cfg_cnn[0][0], cfg_cnn[0][0],
                                         kernel_size=cfg_cnn[0][4], stride=cfg_cnn[0][2],
                                         padding=cfg_cnn[0][3], dilation=cfg_cnn[0][5],
                                         groups=cfg_cnn[0][0])
        nn.init.kaiming_normal_(self.conv1_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv1_pointwise = nn.Conv1d(cfg_cnn[0][0], cfg_cnn[0][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv1_pointwise.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(cfg_cnn[0][1])
        self.activation1 = nn.Hardswish()

        self.conv2_depthwise = nn.Conv1d(cfg_cnn[1][0], cfg_cnn[1][0],
                                         kernel_size=cfg_cnn[1][4], stride=cfg_cnn[1][2],
                                         padding=cfg_cnn[1][3], dilation=cfg_cnn[1][5],
                                         groups=cfg_cnn[1][0])
        nn.init.kaiming_normal_(self.conv2_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv2_pointwise = nn.Conv1d(cfg_cnn[1][0], cfg_cnn[1][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv2_pointwise.weight, mode='fan_out', nonlinearity='relu')
        # self.bn2 = nn.BatchNorm1d(cfg_cnn[1][1])
        self.activation2 = nn.Hardswish()

        self.conv3_depthwise = nn.Conv1d(cfg_cnn[2][0], cfg_cnn[2][0],
                                         kernel_size=cfg_cnn[2][4], stride=cfg_cnn[2][2],
                                         padding=cfg_cnn[2][3], dilation=cfg_cnn[2][5],
                                         groups=cfg_cnn[2][0])
        nn.init.kaiming_normal_(self.conv3_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv3_pointwise = nn.Conv1d(cfg_cnn[2][0], cfg_cnn[2][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv3_pointwise.weight, mode='fan_out', nonlinearity='relu')
        # self.bn3 = nn.BatchNorm1d(cfg_cnn[2][1])
        self.activation3 = nn.Hardswish()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg_cnn[2][1], num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1_depthwise(x)
        x = self.conv1_pointwise(x)
        x = x.unsqueeze(2)
        x = self.bn1(x)
        x = x.squeeze(2)
        x = self.activation1(x)
        x = self.pool(x)
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.activation2(x)
        x = self.pool(x)
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.activation3(x)
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x
    
class NET_1k_MNIST(nn.Module):
    def __init__(self, num_classes):
        super(NET_1k_MNIST, self).__init__()

        cfg_cnn = [(1, 2, 1, 1, 3, 2), (2, 4, 1, 1, 3, 2),(4, 5, 1, 1, 2, 2)]
        
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.conv1_depthwise = nn.Conv2d(cfg_cnn[0][0], cfg_cnn[0][0],
                                         kernel_size=cfg_cnn[0][4], stride=cfg_cnn[0][2],
                                         padding=cfg_cnn[0][3], dilation=cfg_cnn[0][5],
                                         groups=cfg_cnn[0][0])
        self.conv1_pointwise = nn.Conv2d(cfg_cnn[0][0], cfg_cnn[0][1], kernel_size=1)
        self.bn1 = nn.BatchNorm2d(cfg_cnn[0][1])
        self.activation1 = nn.Hardswish()

        self.conv2_depthwise = nn.Conv2d(cfg_cnn[1][0], cfg_cnn[1][0],
                                         kernel_size=cfg_cnn[1][4], stride=cfg_cnn[1][2],
                                         padding=cfg_cnn[1][3], dilation=cfg_cnn[1][5],
                                         groups=cfg_cnn[1][0])
        self.conv2_pointwise = nn.Conv2d(cfg_cnn[1][0], cfg_cnn[1][1], kernel_size=1)
        self.bn2 = nn.BatchNorm2d(cfg_cnn[1][1])
        self.activation2 = nn.Hardswish()

        self.conv3_depthwise = nn.Conv2d(cfg_cnn[2][0], cfg_cnn[2][0],
                                         kernel_size=cfg_cnn[2][4], stride=cfg_cnn[2][2],
                                         padding=cfg_cnn[2][3], dilation=cfg_cnn[2][5],
                                         groups=cfg_cnn[2][0])
        self.conv3_pointwise = nn.Conv2d(cfg_cnn[2][0], cfg_cnn[2][1], kernel_size=1)
        self.bn3 = nn.BatchNorm2d(cfg_cnn[2][1])
        self.activation3 = nn.Hardswish()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg_cnn[2][1], num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1_depthwise(x)
        x = self.conv1_pointwise(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool(x)
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool(x)
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.bn3(x)
        x = self.activation3(x)
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x
    
class NET_4k_ECG(nn.Module):
    def __init__(self, num_classes):
        super(NET_4k_ECG, self).__init__()

        cfg_cnn = [(1, 10, 1, 4, 7, 2),(10, 10, 1, 1, 5, 1),(10, 10, 1, 1, 3, 1)]

        self.quant = torch.quantization.QuantStub() 
        self.dequant = torch.quantization.DeQuantStub()

        # Define layers based on configuration
        self.conv1_depthwise = nn.Conv1d(cfg_cnn[0][0], cfg_cnn[0][0],
                                         kernel_size=cfg_cnn[0][4], stride=cfg_cnn[0][2],
                                         padding=cfg_cnn[0][3], dilation=cfg_cnn[0][5],
                                         groups=cfg_cnn[0][0])
        nn.init.kaiming_normal_(self.conv1_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv1_pointwise = nn.Conv1d(cfg_cnn[0][0], cfg_cnn[0][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv1_pointwise.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(cfg_cnn[0][1])
        self.activation1 = nn.Hardswish()

        self.conv2_depthwise = nn.Conv1d(cfg_cnn[1][0], cfg_cnn[1][0],
                                         kernel_size=cfg_cnn[1][4], stride=cfg_cnn[1][2],
                                         padding=cfg_cnn[1][3], dilation=cfg_cnn[1][5],
                                         groups=cfg_cnn[1][0])
        nn.init.kaiming_normal_(self.conv2_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv2_pointwise = nn.Conv1d(cfg_cnn[1][0], cfg_cnn[1][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv2_pointwise.weight, mode='fan_out', nonlinearity='relu')
        # self.bn2 = nn.BatchNorm1d(cfg_cnn[1][1])
        self.activation2 = nn.Hardswish()

        self.conv3_depthwise = nn.Conv1d(cfg_cnn[2][0], cfg_cnn[2][0],
                                         kernel_size=cfg_cnn[2][4], stride=cfg_cnn[2][2],
                                         padding=cfg_cnn[2][3], dilation=cfg_cnn[2][5],
                                         groups=cfg_cnn[2][0])
        nn.init.kaiming_normal_(self.conv3_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv3_pointwise = nn.Conv1d(cfg_cnn[2][0], cfg_cnn[2][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv3_pointwise.weight, mode='fan_out', nonlinearity='relu')
        # self.bn3 = nn.BatchNorm1d(cfg_cnn[2][1])
        self.activation3 = nn.Hardswish()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg_cnn[2][1], num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1_depthwise(x)
        x = self.conv1_pointwise(x)
        x = x.unsqueeze(2)
        x = self.bn1(x)
        x = x.squeeze(2)
        x = self.activation1(x)
        x = self.pool(x)
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.activation2(x)
        x = self.pool(x)
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.activation3(x)
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

class NET_4k_GSC(nn.Module):
    def __init__(self, num_classes):
        super(NET_4k_GSC, self).__init__()

        cfg_cnn = [(1, 10, 1, 1, 5, 2), (10, 10, 1, 1, 3, 2), (10, 10, 1, 1, 3, 1)] 
        
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.conv1_depthwise = nn.Conv2d(cfg_cnn[0][0], cfg_cnn[0][0],
                                         kernel_size=cfg_cnn[0][4], stride=cfg_cnn[0][2],
                                         padding=cfg_cnn[0][3], dilation=cfg_cnn[0][5],
                                         groups=cfg_cnn[0][0])
        self.conv1_pointwise = nn.Conv2d(cfg_cnn[0][0], cfg_cnn[0][1], kernel_size=1)
        self.bn1 = nn.BatchNorm2d(cfg_cnn[0][1])
        self.activation1 = nn.Hardswish()

        self.conv2_depthwise = nn.Conv2d(cfg_cnn[1][0], cfg_cnn[1][0],
                                         kernel_size=cfg_cnn[1][4], stride=cfg_cnn[1][2],
                                         padding=cfg_cnn[1][3], dilation=cfg_cnn[1][5],
                                         groups=cfg_cnn[1][0])
        self.conv2_pointwise = nn.Conv2d(cfg_cnn[1][0], cfg_cnn[1][1], kernel_size=1)
        self.bn2 = nn.BatchNorm2d(cfg_cnn[1][1])
        self.activation2 = nn.Hardswish()

        self.conv3_depthwise = nn.Conv2d(cfg_cnn[2][0], cfg_cnn[2][0],
                                         kernel_size=cfg_cnn[2][4], stride=cfg_cnn[2][2],
                                         padding=cfg_cnn[2][3], dilation=cfg_cnn[2][5],
                                         groups=cfg_cnn[2][0])
        self.conv3_pointwise = nn.Conv2d(cfg_cnn[2][0], cfg_cnn[2][1], kernel_size=1)
        self.bn3 = nn.BatchNorm2d(cfg_cnn[2][1])
        self.activation3 = nn.Hardswish()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg_cnn[2][1], num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1_depthwise(x)
        x = self.conv1_pointwise(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool(x)
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool(x)
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.bn3(x)
        x = self.activation3(x)
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

class NET_4k_MNIST(nn.Module):
    def __init__(self, num_classes):
        super(NET_4k_MNIST, self).__init__()

        cfg_cnn = [(1, 8, 1, 1, 3, 2), (8, 10, 1, 1, 3, 2), (10, 12, 1, 1, 3, 1)]

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.conv1_depthwise = nn.Conv2d(cfg_cnn[0][0], cfg_cnn[0][0],
                                         kernel_size=cfg_cnn[0][4], stride=cfg_cnn[0][2],
                                         padding=cfg_cnn[0][3], dilation=cfg_cnn[0][5],
                                         groups=cfg_cnn[0][0])
        self.conv1_pointwise = nn.Conv2d(cfg_cnn[0][0], cfg_cnn[0][1], kernel_size=1)
        self.bn1 = nn.BatchNorm2d(cfg_cnn[0][1])
        self.activation1 = nn.Hardswish()

        self.conv2_depthwise = nn.Conv2d(cfg_cnn[1][0], cfg_cnn[1][0],
                                         kernel_size=cfg_cnn[1][4], stride=cfg_cnn[1][2],
                                         padding=cfg_cnn[1][3], dilation=cfg_cnn[1][5],
                                         groups=cfg_cnn[1][0])
        self.conv2_pointwise = nn.Conv2d(cfg_cnn[1][0], cfg_cnn[1][1], kernel_size=1)
        self.bn2 = nn.BatchNorm2d(cfg_cnn[1][1])
        self.activation2 = nn.Hardswish()

        self.conv3_depthwise = nn.Conv2d(cfg_cnn[2][0], cfg_cnn[2][0],
                                         kernel_size=cfg_cnn[2][4], stride=cfg_cnn[2][2],
                                         padding=cfg_cnn[2][3], dilation=cfg_cnn[2][5],
                                         groups=cfg_cnn[2][0])
        self.conv3_pointwise = nn.Conv2d(cfg_cnn[2][0], cfg_cnn[2][1], kernel_size=1)
        self.bn3 = nn.BatchNorm2d(cfg_cnn[2][1])
        self.activation3 = nn.Hardswish()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(cfg_cnn[2][1], num_classes)

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1_depthwise(x)
        x = self.conv1_pointwise(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.pool(x)
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.pool(x)
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.bn3(x)
        x = self.activation3(x)
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

class NET_32k_ECG(nn.Module):
    def __init__(self, num_classes):
        super(NET_32k_ECG, self).__init__()

        # Model Definition
        cfg_cnn = [(1, 40, 1, 4, 7, 2), (40, 40, 1, 1, 5, 1), (40, 40, 1, 1, 3, 1)]

        self.quant = torch.quantization.QuantStub() 
        self.dequant = torch.quantization.DeQuantStub()

        # Define layers based on configuration
        self.conv1_depthwise = nn.Conv1d(cfg_cnn[0][0], cfg_cnn[0][0],
                                         kernel_size=cfg_cnn[0][4], stride=cfg_cnn[0][2],
                                         padding=cfg_cnn[0][3], dilation=cfg_cnn[0][5],
                                         groups=cfg_cnn[0][0])
        nn.init.kaiming_normal_(self.conv1_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv1_pointwise = nn.Conv1d(cfg_cnn[0][0], cfg_cnn[0][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv1_pointwise.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(cfg_cnn[0][1])
        self.activation1 = nn.Hardswish()

        self.conv2_depthwise = nn.Conv1d(cfg_cnn[1][0], cfg_cnn[1][0],
                                         kernel_size=cfg_cnn[1][4], stride=cfg_cnn[1][2],
                                         padding=cfg_cnn[1][3], dilation=cfg_cnn[1][5],
                                         groups=cfg_cnn[1][0])
        nn.init.kaiming_normal_(self.conv2_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv2_pointwise = nn.Conv1d(cfg_cnn[1][0], cfg_cnn[1][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv2_pointwise.weight, mode='fan_out', nonlinearity='relu')
        # self.bn2 = nn.BatchNorm1d(cfg_cnn[1][1])
        self.activation2 = nn.Hardswish()

        self.conv3_depthwise = nn.Conv1d(cfg_cnn[2][0], cfg_cnn[2][0],
                                         kernel_size=cfg_cnn[2][4], stride=cfg_cnn[2][2],
                                         padding=cfg_cnn[2][3], dilation=cfg_cnn[2][5],
                                         groups=cfg_cnn[2][0])
        nn.init.kaiming_normal_(self.conv3_depthwise.weight, mode='fan_out', nonlinearity='relu')
        self.conv3_pointwise = nn.Conv1d(cfg_cnn[2][0], cfg_cnn[2][1], kernel_size=1)
        nn.init.kaiming_normal_(self.conv3_pointwise.weight, mode='fan_out', nonlinearity='relu')
        # self.bn3 = nn.BatchNorm1d(cfg_cnn[2][1])
        self.activation3 = nn.Hardswish()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(cfg_cnn[2][1], num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1_depthwise(x)
        x = self.conv1_pointwise(x)
        x = x.unsqueeze(2)
        x = self.bn1(x)
        x = x.squeeze(2)
        x = self.activation1(x)
        x = self.pool(x)
        x = self.conv2_depthwise(x)
        x = self.conv2_pointwise(x)
        x = self.activation2(x)
        x = self.pool(x)
        x = self.conv3_depthwise(x)
        x = self.conv3_pointwise(x)
        x = self.activation3(x)
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

class NET_32k_MNIST(nn.Module):
    def __init__(self, num_classes):
        super(NET_32k_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)  # Batch normalization layer
        self.activation1 = nn.Hardswish()  # Hardswish activation function

        self.conv2 = nn.Conv2d(10, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)  # Batch normalization layer
        self.activation2 = nn.Hardswish()  # Hardswish activation function

        self.conv3 = nn.Conv2d(12, 14, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(14)  # Batch normalization layer
        self.activation3 = nn.Hardswish()  # Hardswish activation function

        self.fc = nn.Linear(14 * 3 * 3, num_classes)  # Adjust according to the output of the convolutional layers

        self.quant = torch.quantization.QuantStub()  # Input quantization node
        self.dequant = torch.quantization.DeQuantStub()  # Output dequantization node

    def forward(self, x):
        x = self.quant(x)
        x = self.activation1(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.activation2(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.activation3(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

class NET_32k_GSC(nn.Module):
    def __init__(self, num_classes):
        super(NET_32k_GSC, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.activation1 = nn.Hardswish()

        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.activation2 = nn.Hardswish()

        self.conv3 = nn.Conv2d(8, 12, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(12) 
        self.activation3 = nn.Hardswish()

        self.fc = nn.Linear(12 * 4 * 4, num_classes)

        self.quant = torch.quantization.QuantStub() 
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.activation1(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.activation2(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.activation3(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

class NET_32k_MNIST(nn.Module):
    def __init__(self, num_classes):
        super(NET_32k_MNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)  # Batch normalization layer
        self.activation1 = nn.Hardswish()  # Hardswish activation function

        self.conv2 = nn.Conv2d(10, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)  # Batch normalization layer
        self.activation2 = nn.Hardswish()  # Hardswish activation function

        self.conv3 = nn.Conv2d(12, 14, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(14)  # Batch normalization layer
        self.activation3 = nn.Hardswish()  # Hardswish activation function

        self.fc = nn.Linear(14 * 3 * 3, num_classes)  # Adjust according to the output of the convolutional layers

        self.quant = torch.quantization.QuantStub()  # Input quantization node
        self.dequant = torch.quantization.DeQuantStub()  # Output dequantization node

    def forward(self, x):
        x = self.quant(x)
        x = self.activation1(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.activation2(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.activation3(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)  # Note: Apply dequantization before output
        return x

# MultiChannelTimeSeriesNet for Motion State Detection (Fig4) 

# (channels, stride, padding, kernel_size, dilation)
cfg_NET_1k_MSD = {
    'channel1': [( 3, 1, 1, 3, 1), ( 2, 1, 1, 3, 1)],
    'channel2': [( 3, 1, 1, 3, 1), ( 2, 1, 1, 3, 1)],
    'channel3': [( 2, 1, 1, 3, 1), ( 2, 1, 1, 2, 1)],
    'channel4': [( 2, 1, 1, 3, 1), ( 2, 1, 1, 2, 1)]
}

class NET_1k_DAM(nn.Module):
    def __init__(self, num_classes=5, gap_size=1, cfg=cfg_NET_1k_MSD):
        super(NET_1k_DAM, self).__init__()

        self.channel1_conv = self._create_conv_block(cfg['channel1'])
        self.channel2_conv = self._create_conv_block(cfg['channel2'])
        self.channel3_conv = self._create_conv_block(cfg['channel3'])
        self.channel4_conv = self._create_conv_block(cfg['channel4'])

        self.gap1 = nn.AdaptiveAvgPool1d(gap_size)
        self.gap2 = nn.AdaptiveAvgPool1d(gap_size)
        self.gap3 = nn.AdaptiveAvgPool1d(gap_size)
        self.gap4 = nn.AdaptiveAvgPool1d(gap_size)

        last_channels = cfg['channel1'][-1][0] + cfg['channel2'][-1][0] + cfg['channel3'][-1][0] + cfg['channel4'][-1][0]
        fc_input_size = last_channels * gap_size

        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, num_classes),
        )

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()


    def _create_conv_block(self, channels):
        layers = []
        in_channels = 1

        for out_channels, stride, padding, kernel_size, dilation in channels:
            conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding, dilation=dilation)
            init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            
            layers.append(conv)
            layers.append(nn.ReLU())
            in_channels = out_channels

        return nn.Sequential(*layers)
    
    def forward(self, x1, x2, x3, x4):
        # 删除NaN部分
        x1 = self._remove_nan(x1)
        x2 = self._remove_nan(x2)
        x3 = self._remove_nan(x3)
        x4 = self._remove_nan(x4)

        x1 = self.quant(x1)
        x2 = self.quant(x2)
        x3 = self.quant(x3)
        x4 = self.quant(x4)

        x1 = self.channel1_conv(x1)
        x2 = self.channel2_conv(x2)
        x3 = self.channel3_conv(x3)
        x4 = self.channel4_conv(x4)

        x1 = self.gap1(x1)
        x2 = self.gap2(x2)
        x3 = self.gap3(x3)
        x4 = self.gap4(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)

        return x

    def _remove_nan(self, x):
        non_nan_indices = ~torch.isnan(x[0, 0, 0])
        x = x[:, :, :, non_nan_indices]
        return x.squeeze(2)


