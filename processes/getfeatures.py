import torch
import torch.nn.functional as F
import os
import sys
sys.path.append('/../arcface_model')

print(sys.path)
from arcface_model.iresnet import iresnet100


def get_zid(img):
    # Is input size 256 or 224? 일단 모두 256으로 가정
    netArc = iresnet100(fp16=False)
    print(os.getcwd())
    netArc.load_state_dict(torch.load('../arcface_model/backbone.pth'))
    netArc=netArc.cuda()
    netArc.eval()

    # get the identity embeddings of X_s
    with torch.no_grad():
        z_id = netArc(F.interpolate(img, [112, 112], mode='bilinear', align_corners=False))

    return z_id