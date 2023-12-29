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

        #print("innerproduct shape before squeeze :", torch.mul(self.z_id_yhat_st, self.z_id_x_s).shape)
        #inner_product = (torch.mul(self.z_id_yhat_st, self.z_id_x_s).squeeze()) # -1 stands for batch size / Being batch fixed, matrix multiplication
        cos_sim = torch.cosine_similarity(self.z_id_yhat_st, self.z_id_x_s, dim=1)
        #print("innerproduct shape :", inner_product.shape)
        return self.l1(torch.ones_like(cos_sim), cos_sim)
    
    def chgLoss(self):
        return self.l1(self.yhat_st, self.y_st)
    
    def recLoss(self):
        L_rec = torch.sum(0.5 * torch.mean(torch.pow(self.y_st - self.x_t, 2).reshape(self.y_st.shape[0], -1), dim=1) * self.same ) / (self.same.sum() + 1e-6)
        return L_rec
    
    def hearnetLoss(self):
        return self.idLoss() + self.chgLoss() + self.recLoss() # Coefficients are all zeros in the paper