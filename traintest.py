from argparse import ArgumentParser
import torch
import sys, os
from models.hearnet import HearNet
from losses.loss import HearNetLoss
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.append('/workspace')
from processes.getfeatures import get_zid


import torch
from torch.utils.data import Dataset, DataLoader
# Transform function defined
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Image resize just in case
    transforms.ToTensor(),           # Image to tensor
])

source_path = './data/source'
source_collection = ImageFolder(root=source_path, transform=transform)

print(iter(source_collection.shape))