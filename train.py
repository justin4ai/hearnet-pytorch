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

sys.path.append('/workspace')
from processes.getfeatures import get_zid


import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, source_collection, target_collection, yhat_st_collection, h_error_collection, batch_size):

        src_iter, tgt_iter, swapped_iter, heuristic_iter = iter(source_collection), iter(target_collection), iter(yhat_st_collection), iter(h_error_collection)
        
        srcList, tgtList, swappedList, heuristicList = [], [], [], []

        while True:
            try :
                src, tgt, swapped, heuristic = next(src_iter)[0], next(tgt_iter)[0], next(swapped_iter)[0], next(heuristic_iter)[0]
                srcList.append(src)
                tgtList.append(tgt)
                swappedList.append(swapped)
                heuristicList.append(heuristic)

            except StopIteration:
                break
        #self.source_collection = iter(source_collection)
        self.source_collection = torch.cat(srcList, dim=0)
        print(self.source_collection.shape)
        self.target_collection = torch.cat(tgtList, dim=0)
        self.yhat_st_collection = torch.cat(swappedList, dim=0)
        self.h_error_collection = torch.cat(heuristicList, dim=0)
        #self.source_collection = source_collection
        # self.target_collection = target_collection
        # self.yhat_st_collection = yhat_st_collection
        # self.h_error_collection = h_error_collection
        self.batch_size = batch_size
        self.num_samples = len(yhat_st_collection)  # Assume all the legnths are equal

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        #start_idx = index * self.batch_size
        #end_idx = (index + 1) * self.batch_size
        #print("No prob")
        #print(type(self.source_collection))
        #print(next(self.source_collection)[0].shape)
        source_batch = self.source_collection[index]
        #print("No prob2")
        target_batch = self.target_collection[index]
        yhat_st_batch = self.yhat_st_collection[index]
        h_error_batch = self.h_error_collection[index]

        return source_batch, target_batch, yhat_st_batch, h_error_batch

def main(args):

    device = args.device

    # Data paths
    source_path = args.source_images # Assuming (10000 x 3 x 256 x 256)
    target_path = args.target_images # Assuming (10000 x 3 x 256 x 256)
    yhat_st_path = args.swapped_images # Assuming (10000 x 3 x 256 x 256)
    h_error_path = args.heuristic_errors # Assuming (10000 x 3 x 256 x 256)
    


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

    print(source_collection.classes)
    print(source_collection.class_to_idx)

    # configs
    batch_size = 2
    in_channels = 6
    out_channels = 3
    num_epochs = 10 

    # Data loader defined
    my_dataset = MyDataset(source_collection, target_collection, yhat_st_collection, h_error_collection, batch_size)
    train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)



    # Model object created
    hear_net = HearNet(in_channels, out_channels)
    optimizer = optim.Adam(hear_net.parameters(), lr=0.001)


    for epoch in range(num_epochs):
        hear_net.train()
        
        for batch_idx, (x_s, x_t, yhat_st, h_error) in enumerate(train_loader): # with batch size
            print("Reaching this line successfully"
            )
            optimizer.zero_grad()
            
            print(f"first shape : {x_s.shape}")

            print(f"second shape : {x_t.shape}")
            print(f"third shape : {yhat_st.shape}")
            print(f"fourth shape : {h_error.shape}")
            # Move to GPU
            x_s = x_s.to(device)
            x_t = x_t.to(device)
            yhat_st = yhat_st.to(device)
            h_error = h_error.to(device)

            # Forward pass
            output = hear_net(yhat_st, h_error)
            
            # Compute the loss
            z_id_yhat_st = get_zid(yhat_st) # Extracted from ghost module
            z_id_x_s = get_zid(x_s)

            hear_net_loss = HearNetLoss(z_id_yhat_st, z_id_x_s, output, yhat_st, x_s, x_t) # output argument goes to y_st parameter
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
    parser.add_argument("--source_images", default='./data/source', help="")
    parser.add_argument("--target_images", default='./data/target', help="")
    parser.add_argument("--swapped_images", default='./data/swapped', help="")
    parser.add_argument("--heuristic_errors", default='./data/heuristic', help="")
    parser.add_argument("--save_path", default='./checkpoints/hearnet/hear_net.pth', help="")
    
    parser.add_argument("--cpu", default=False)
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda" # Use export CUDA_VISIBLE_DEVICES=? in terminal
    else:
        args.device = "cpu"

    main(args)