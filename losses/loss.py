import torch
import torch.nn as nn
import torch.nn.functional as F

class HearNetLoss():

    def __init__(self, zIdEncoder, y_st, yhat_st, x_s, x_t):
        self.identityEncoder = 'PLACEHOLDER'
        self.y_st = 'PLACEHOLDER'
        self.yhat_st = 'PLACEHOLDER'
        self.x_s = 'PLACEHOLDER'
        self.x_t = 'PLACEHOLDER'

        


    def idLoss(self, y_st, x_s):
        import math

        return 1 - math.cos(self.identityEncoder(y_st), self.identityEncoder(x_s))
    
    def chgLoss(self, yhat_st, y_st):
        return abs(yhat_st - y_st) # Pixel-wise subtraction needed
    
    def recLoss(self, y_st, x_t, x_s):
        
        return 0.5*(y_st - x_t)^2 if (x_t == x_s) else 0
    
    def hearnetLoss(self):
        return self.idLoss() + self.chgLoss() + self.recLoss()