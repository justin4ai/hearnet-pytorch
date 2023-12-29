from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torchvision
import numpy as np


from models.hearnet import HearNet
from argparse import ArgumentParser
from processes.getfeatures import get_zid
from losses.loss import HearNetLoss
import random

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    print(img.shape)
    
    npimg = img.cpu().numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

class MyDataset(Dataset):
    def __init__(self, source_collection, target_collection, yhat_st_collection, h_error_collection, batch_size, same_prob=0.8, same_identity=False):

        self.same_prob = same_prob
        self.same_identity = same_identity
        
        source_iter = iter(source_collection)
        target_iter = iter(target_collection)
        yhat_st_iter = iter(yhat_st_collection)
        h_error_iter = iter(h_error_collection)
        
        source_list = []
        target_list = []
        yhat_st_list = []
        h_error_list = []

        while True:
            try :
                source = next(source_iter)[0]
                target = next(target_iter)[0]
                yhat_st = next(yhat_st_iter)[0]
                h_error = next(h_error_iter)[0]
                
                source_list.append(source)
                target_list.append(target)
                yhat_st_list.append(yhat_st)
                h_error_list.append(h_error)
                
            except StopIteration:
                break
        
        self.source_collection = torch.cat(source_list, dim=0)
        self.target_collection = torch.cat(target_list, dim=0)
        self.yhat_st_collection = torch.cat(yhat_st_list, dim=0)
        self.h_error_collection = torch.cat(h_error_list, dim=0)
        
        self.batch_size = batch_size
        self.num_samples = len(yhat_st_collection)  # Assume all the legnths are equal

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        source_batch = self.source_collection[index : index + 3, :, :]
        target_batch = self.target_collection[index : index + 3, :, :]
        yhat_st_batch = self.yhat_st_collection[index : index + 3, :, :]
        h_error_batch = self.h_error_collection[index : index + 3, :, :]
        
        if random.random() > self.same_prob:
            same_person = 0
        else:
            target_batch = source_batch
            same_person = 1

        return source_batch, target_batch, yhat_st_batch, h_error_batch, same_person


def train(args, device):
    
    #path
    source_path = args.source_images
    target_path = args.target_images
    yhat_st_path = args.yhat_st_images
    h_error_path = args.h_error_images
    
    batch_size = args.batch_size
    
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
    
    my_dataset = MyDataset(source_collection, target_collection, yhat_st_collection, h_error_collection, batch_size)
    train_loader = DataLoader(my_dataset, batch_size=batch_size, shuffle=False)
    
    #configs
    in_channels = 6
    out_channels = 3
    num_epochs = 3
    
    hear_net = HearNet(in_channels, out_channels)
    hear_net.to(device)
    hear_net.train()
    optimizer = optim.Adam(hear_net.parameters(), lr=0.001)
    
    writer = SummaryWriter('runs/test2')

    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        
        for x_s, x_t, yhat_st, h_error, same in train_loader:
            # x_s = torch.tensor(x_s)
            # x_t = torch.tensor(x_t)
            # yhat_st = torch.tensor(yhat_st)
            # h_error = torch.tensor(h_error)
            x_s = x_s.clone().detach()
            x_t = x_t.clone().detach()
            yhat_st = yhat_st.clone().detach()
            h_error = h_error.clone().detach()


            x_s = x_s.to(device)
            x_t = x_t.to(device)
            yhat_st = yhat_st.to(device)
            h_error = h_error.to(device)
            same = same.to(device)
            
            output = hear_net(yhat_st, h_error)
            
            z_id_yhat_st = get_zid(yhat_st) # Extracted from ghost module
            z_id_x_s = get_zid(x_s)

            hear_net_loss = HearNetLoss(z_id_yhat_st, z_id_x_s, output, yhat_st, x_s, x_t, same) # output argument goes to y_st parameter
            loss = hear_net_loss.hearnetLoss()
            
            print(output.shape)
            img_grid = torchvision.utils.make_grid(output)
            #matplotlib_imshow(img_grid, one_channel=True)

            
            # writer.add_image('four_fashion_mnist_images', img_grid)

            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()*output.size(0)
            
        writer.add_images('Output Images', output, global_step=global_step, dataformats='NCHW')
        global_step += 1
        #if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(my_dataset)}')
            
    torch.save(hear_net.state_dict(), args.save_path)



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print('cuda is not available. using cpu. check if it\'s ok')
    print("Starting traing")
    train(args, device=device)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--yhat_st_images', default='./datasets/yhat_st_images')
    parser.add_argument('--source_images', default='./datasets/source_images')
    parser.add_argument('--target_images', default='./datasets/target_images')
    parser.add_argument('--h_error_images', default='./datasets/h_error_images')
    parser.add_argument('--save_path', default='./checkpoints/hearnet_model.pth')
    
    args = parser.parse_args()

    main(args)