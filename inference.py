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


def main(args):
    #torch.backends.cudnn.enabled = False

    hearnet_ckpt = args.hearnet_ckpt
    current_root_path = os.path.split(sys.argv[0])[0]
    model = HearNet()
    model.load_state_dict(torch.load(hearnet_ckpt))
    model.eval()

if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--hearnet_ckpt", default='', help="")
    
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda" # Use export CUDA_VISIBLE_DEVICES=? in terminal
    else:
        args.device = "cpu"

    main(args)