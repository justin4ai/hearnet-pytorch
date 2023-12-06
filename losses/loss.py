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

        self.l1 = nn.L1Loss()

        


    def idLoss(self, y_st, x_s):
        inner_product = (torch.bmm(y_st.view(-1, 1, config.z_id_size), x_s.view(-1, config.z_id_size, 1)).squeeze())
        return self.l1(torch.ones_like(inner_product), inner_product)
    
    def chgLoss(self, yhat_st, y_st):
        return self.l1(yhat_st - y_st) # Automatic pixel-wise subtraction
    
    def recLoss(self, y_st, x_t, x_s):
        
        return 0.5*(y_st - x_t)^2 if (x_t == x_s) else 0
    
    def hearnetLoss(self):
        return self.idLoss() + self.chgLoss() + self.recLoss()