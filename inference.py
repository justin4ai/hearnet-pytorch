from argparse import ArgumentParser
import torch
import sys, os

def main(args):
    #torch.backends.cudnn.enabled = False

    checkpoint_path = args.checkpoints

    current_root_path = os.path.split(sys.argv[0])[0]


if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--checkpoints", default='', help="")
    
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda" # Use export CUDA_VISIBLE_DEVICES=? in terminal
    else:
        args.device = "cpu"

    main(args)