import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
from torch.utils.data import DataLoader, Subset
import torchvision
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay