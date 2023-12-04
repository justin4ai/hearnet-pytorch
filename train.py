from argparse import ArgumentParser
import torch
import sys, os
from models.hearnet import HearNet
from losses.loss import HearNetLoss
import torch
import torch.nn as nn
import torch.optim as optim

def main(args):
    #torch.backends.cudnn.enabled = False

    data_path = args.data_path

    current_root_path = os.path.split(sys.argv[0])[0]

    # Assuming you have your data loaders (train_loader, val_loader) and the necessary components initialized
    in_channels = 3
    out_channels = 3

    hear_net = HearNet(in_channels, out_channels)
    identity_encoder = 'PLACEHOLDER'  # Gotta be entangled with ghost module
    hear_net_loss = HearNetLoss(identity_encoder, 'PLACEHOLDER', 'PLACEHOLDER', 'PLACEHOLDER', 'PLACEHOLDER')  # Replace with your y_st value

    optimizer = optim.Adam(hear_net.parameters(), lr=0.001)


    num_epochs = 10 
    bs ='PLACEHOLDER' # Batch size will be added
    train_loader = 'PLACEHOLDER' # train_loader will be added
    

    for epoch in range(num_epochs): # Tqdm will be added
        hear_net.train()
        
        for batch_idx, (yhat_st_batch, h_error_batch) in enumerate(train_loader):  
            optimizer.zero_grad()
            
            # Forward pass
            output = hear_net(yhat_st_batch, h_error_batch)
            
            # Compute the loss
            #hear_net_loss.yhat_st = output  # Update yhat_st in HearNetLoss
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

    # Save the trained model if needed
    torch.save(hear_net.state_dict(), 'hear_net_model.pth')


if __name__ == '__main__':

    parser = ArgumentParser()  
    parser.add_argument("--data_path", default='', help="")
    
    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda" # Use export CUDA_VISIBLE_DEVICES=? in terminal
    else:
        args.device = "cpu"

    main(args)