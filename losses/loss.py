import torch
import torch.nn as nn
import torch.nn.functional as F

class HearNetLoss():

    def __init__(self, z_id_yhat_st, z_id_x_s, y_st, yhat_st, x_s, x_t, same):
        self.z_id_yhat_st = z_id_yhat_st
        self.z_id_x_s = z_id_x_s
        self.y_st = y_st
        self.yhat_st = yhat_st
        self.x_s = x_s
        self.x_t = x_t
        self.same = same

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()


    def idLoss(self):
        inner_product = (torch.matmul(self.z_id_yhat_st.view(-1, 1, 512), self.z_id_x_s.view(-1, 1, 512)).squeeze()) # -1 stands for batch size / Being batch fixed, matrix multiplication
        return self.l1(torch.ones_like(inner_product), inner_product)
    
    def chgLoss(self):
        return self.l1(self.yhat_st, self.y_st)
    
    def recLoss(self):
        '''
        same : Having (batch_size) shape (1 dimensional). Represents whether source and target are the same (binary). 
        '''
        self.same = self.same.unsqueeze(-1).unsqueeze(-1) # Becomes 4 dimesional from 2 dimensional tensor
        self.same = self.same.expand(self.x_t.shape) # Assuming x_t and y_st have the same shape. Becomes one_likes or zero_likes.
        self.x_t = torch.mul(self.x_t, self.same) # Being batch and channel fixed, matrix multiplication
        self.y_st = torch.mul(self.y_st, self.same) # Being batch and channel fixed, matrix multiplication
        return 0.5 * self.l2(self.x_t, self.y_st)
    
    def hearnetLoss(self):
        return self.idLoss() + self.chgLoss() + self.recLoss() # Coefficients are all zeros in the paper