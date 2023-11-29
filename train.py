from argparse import ArgumentParser
import torch
import sys, os

def main(args):
    #torch.backends.cudnn.enabled = False

    data_path = args.data_path

    current_root_path = os.path.split(sys.argv[0])[0]


if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--data_path", default='', help="")
    
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda" # Use export CUDA_VISIBLE_DEVICES=? in terminal
    else:
        args.device = "cpu"

    main(args)