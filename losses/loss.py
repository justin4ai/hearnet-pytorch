import torch
import torch.nn as nn
import torch.nn.functional as F

class HearNetLoss():

    def __init__(self, z_id, y_st, yhat_st, x_s, x_t, same):
        self.z_id = z_id
        self.y_st = y_st
        self.yhat_st = yhat_st
        self.x_s = x_s
        self.x_t = x_t
        self.same = same

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()


    def idLoss(self):
        inner_product = (torch.bmm(self.y_st.view(-1, 1, config.z_id_size), self.x_s.view(-1, config.z_id_size, 1)).squeeze()) # Modification needed.
        return self.l1(torch.ones_like(inner_product), inner_product)
    
    def chgLoss(self):
        return self.l1(self.yhat_st, self.y_st)
    
    def rec_loss(self):
        '''
        same : Having (batch_size) shape (1 dimensional). Represents whether source and target are the same (binary). 
        '''
        self.same = self.same.unsqueeze(-1).unsqueeze(-1)
        self.same = self.same.expand(self.x_t.shape) # Assuming x_t and y_st have the same shape. One_likes or zero_likes.
        self.x_t = torch.mul(self.x_t, self.same) # Replace torch.mul with torch.bmm?
        self.y_st = torch.mul(self.y_st, self.same) # Replace torch.mul with torch.bmm?
        return 0.5 * self.l2(self.x_t, self.y_st)
    
    def hearnetLoss(self):
        return self.idLoss() + self.chgLoss() + self.recLoss() # Coefficients are all zeros in the paper