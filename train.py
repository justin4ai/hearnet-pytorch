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
from processes.getfeatures import get_zid

import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, source_collection, target_collection, yhat_st_collection, h_error_collection, batch_size):
        '''
        yhat_st_collection : 
            - This stands for the collection of yhat_st images (swapped by ghost)
            - Shape of (10000(= example, total #(images)), 3(= #(channel)), 256(= width), 256(= height))
        h_error_collection :
            - This stands for the collection of heuristic errors
            - Shape of (10000(= example, total #(images)), 3(= #(channel)), 256(= width), 256(= height))
        '''
        self.source_collection = source_collection
        self.target_collection = target_collection
        self.yhat_st_collection = yhat_st_collection
        self.h_error_collection = h_error_collection
        self.batch_size = batch_size
        self.num_samples = len(yhat_st_collection)  # Assume all the legnths are equal

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size

        source_batch = self.source_collection[start_idx:end_idx]
        target_batch = self.target_collection[start_idx:end_idx]
        yhat_st_batch = self.yhat_st_collection[start_idx:end_idx]
        h_error_batch = self.h_error_collection[start_idx:end_idx]

        return source_batch, target_batch, yhat_st_batch, h_error_batch

def main(args):

    device = args.device

    # Data paths
    source_path = args.source_images # Assuming (10000 x 3 x 256 x 256)
    target_path = args.target_images # Assuming (10000 x 3 x 256 x 256)
    yhat_st_path = args.swapped_images # Assuming (10000 x 3 x 256 x 256)
    h_error_path = args.heuristic_error # Assuming (10000 x 3 x 256 x 256)
    
    # Transform function defined
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Image resize just in case
        transforms.ToTensor(),           # Image to tensor
    ])

    # Load images
    source_collection = ImageFolder(root=source_path, transform=transform)
    target_collection = ImageFolder(root=target_path , transform=transform)
    yhat_st_collection = ImageFolder(root=yhat_st_path, transform=transform)
    h_error_collection = ImageFolder(root=h_error_path , transform=transform)

    # Data loader defined
    my_dataset = MyDataset(source_collection, target_collection, yhat_st_collection, h_error_collection, batch_size) # Batch_size? 
    train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True) # Batch_size?

    # configs
    batch_size = 100
    in_channels = 6
    out_channels = 3
    num_epochs = 10 

    # Model object created
    hear_net = HearNet(in_channels, out_channels)
    optimizer = optim.Adam(hear_net.parameters(), lr=0.001)


    for epoch in range(num_epochs):
        hear_net.train()
        
        for batch_idx, (x_s, x_t, yhat_st, h_error) in enumerate(train_loader): # with batch size
            optimizer.zero_grad()
            
            # Move to GPU
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            yhat_st = yhat_st.to(device)
            h_error = h_error.to(device)

            # Forward pass
            output = hear_net(yhat_st, h_error)
            
            # Compute the loss
            z_id = get_zid() # Extracted from ghost module


            hear_net_loss = HearNetLoss(z_id, output, yhat_st, x_s, x_t) # output argument goes to y_st parameter
            loss = hear_net_loss.hearnetLoss()
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item()}')



        ## Evaluate the model on the validation set if needed
        #hear_net.eval()
        #with torch.no_grad():
        #    for val_yhat_st_batch, val_h_error_batch in val_loader:  # Replace with your validation data loader
        #        val_output = hear_net(val_yhat_st_batch, val_h_error_batch)
        #        # Add any validation-related code here

    # Save the trained model
    torch.save(hear_net.state_dict(), args.save_path)


if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--source_images", default='', help="")
    parser.add_argument("--target_errors", default='', help="")
    parser.add_argument("--swapped_images", default='', help="")
    parser.add_argument("--heuristic_errors", default='', help="")
    parser.add_argument("--save_path", default='./checkpoints/hearnet/hear_net.pth', help="")
    
    parser.add_argument("--cpu", default=False)
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda" # Use export CUDA_VISIBLE_DEVICES=? in terminal
    else:
        args.device = "cpu"

    main(args)