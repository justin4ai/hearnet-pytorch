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
from torchvision.utils import save_image

def main(args):
    #torch.backends.cudnn.enabled = False

    hearnet_ckpt = args.hearnet_ckpt
    #current_root_path = os.path.split(sys.argv[0])[0]
    model = HearNet()
    model.load_state_dict(torch.load(hearnet_ckpt))

    with torch.no_grad():
        model.eval()
        
        output = model()
        save_image(output , args.save_path)

        

        return 

if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--hearnet_ckpt", default='', help="")

    parser.add_argument("--swapped_images", default='./data/swapped', help="")
    parser.add_argument("--heuristic_errors", default='./data/heuristic', help="")

    parser.add_argument("--save_path", default='./data/result/output.png', help="")

    args = parser.parse_args()



    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda" # Use export CUDA_VISIBLE_DEVICES=? in terminal
    else:
        args.device = "cpu"

    main(args)