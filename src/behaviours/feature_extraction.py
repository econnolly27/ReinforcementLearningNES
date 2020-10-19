
# Implements a Neural Network for extracting features from single images for BBRL that can be used as the visual backbone
# for training button behaviours, e.g. behaviour.py. This script includes train and test functions,
# and can be used as standalone. Remember to first generate a dataset with "gencsv_fe.py" to create a csv file with paths.
# Line 217 defines the dataset file to be used
# Authors: Gerardo Aragon-Camarasa, 2020

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
import torchvision.models as models
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# current_path = os.path.abspath('.')
# parent_path = os.path.dirname(current_path)
# sys.path.append(parent_path) 

try:
    from behaviours import mariodataloader as md
    from behaviours import plot
except:
    import mariodataloader as md
    import plot

from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim
import time
import pickle
from torch.optim.lr_scheduler import StepLR

img_size = 256


class FeatureExtraction(nn.Module):
    def __init__(self, device):
        super(FeatureExtraction, self).__init__()
        self._device = device

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # b, 32, 128, 128 = 524,288
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # b, 32, 64, 64 = 131,072
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # b, 64, 32, 32 = 65,536
        # b, 128, 16, 16 = 32,768 -> 16.6% compression
        # b, 64, 16, 16 = 16,384 -> 8.3% compression
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # b, 64, 32, 32
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1) # b, 32, 64, 64
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1) # b, 32, 128, 128
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1) # b, 3, 256, 256

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)
        self.elu = nn.ELU()
    
    def encode(self, x):
        out = self.elu(self.conv1(x))
        out = self.elu(self.conv2(out))
        out = self.elu(self.conv3(out))
        out = self.elu(self.conv4(out))
        return out

    def decode(self, x):
        out = self.elu(self.deconv1(x))
        out = self.elu(self.deconv2(out))
        out = self.elu(self.deconv3(out))
        out = self.elu(self.deconv4(out))
        out = self.tanh(out)
        return out

    def forward(self, x):
        feats = self.encode(x)
        out = self.decode(feats)
        return out

    @staticmethod
    def loss_function(recon_x, x):
        loss = nn.MSELoss(reduction='sum')(recon_x, x)
        return loss

    @staticmethod
    def to_img(x):
        # Function when image is normalised...
        # x = 0.5 * (x + 1)
        # x = x.clamp(0, 1)
        # x = x.view(x.size(0), 3, img_size, img_size)
        return x


def trainNet(net, train_loader, val_loader, n_epochs, learning_rate, device):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    #Time for printing
    training_start_time = time.time()

    save_loss = []
    prev_loss = np.inf

    scheduler = StepLR(optimizer, step_size=8, gamma=0.96)
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        train_loss  = 0.0
        mse_loss = 0.0
        epoch_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        print('==> Epoch:', epoch,'LR:', scheduler.get_last_lr())
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs = data['image'].to(device)
            
            #Wrap them in a Variable object
            inputs = Variable(inputs)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            recon_batch = net(inputs)
            loss = net.loss_function(recon_batch, inputs)
            loss2 = nn.MSELoss(reduction='mean')(recon_batch, inputs)
            loss.backward()            
            optimizer.step()
            
            #Print statistics
            train_loss  += loss.item()
            mse_loss += loss2.item()
            epoch_loss += loss2.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print('Epoch [{}/{}] {:d}%, Step [{}/{}], Loss: {:.4f} (MSE: {:.4f}), took: {:.2f}s'
                  .format(epoch + 1, n_epochs, int(100 * (i+1) / n_batches), i + 1, len(train_loader), train_loss  / print_every, mse_loss / print_every, time.time() - start_time))
                save_dir = './plots'
                if epoch > 0: # Don't plot first epoch
                    plot.plot(save_dir, '/loss trainAE', train_loss  / print_every)
                    plot.flush()
                    plot.tick()
                #Reset running loss and time
                train_loss = 0.0
                mse_loss = 0.0
                start_time = time.time()
            
        epoch_loss_train = epoch_loss / len(train_loader)
        scheduler.step()
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0.0
        total_mse_loss = 0.0
        batch_size = val_loader.batch_size
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):

                #Get inputs
                inputs = data['image'].to(device)
                
                #Wrap tensors in Variables
                inputs = Variable(inputs)
                
                #Forward pass
                net.eval()
                recon_batch = net(inputs)
                total_val_loss += net.loss_function(recon_batch, inputs).item()
                total_mse_loss += nn.MSELoss(reduction='mean')(recon_batch, inputs).item()

                if i == 0:
                    n = min(inputs.size(0), 8)
                    o_img = net.to_img(inputs[:n])
                    recon_img = net.to_img(recon_batch.view(batch_size, 3, img_size, img_size)[:n])
                    comparison = torch.cat([o_img, recon_img])
                    save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch + 1) + '.png', nrow=n)
                
            print("Validation loss = {:.4f}; MSE {:.4f}".format(total_val_loss / len(val_loader), total_mse_loss / len(val_loader)))
            save_loss.append([total_val_loss / len(val_loader), total_mse_loss / len(val_loader)])

        save_dir = './plots'
        plot.plot_vt(save_dir, '/epoch_lossAE', epoch_loss_train, total_mse_loss / len(val_loader))
        plot.flush_vt()
        plot.tick_vt()
        
        # torch.save(net.state_dict(), './models/modelv5_epoch_' + str(epoch) + '.pth')
        if prev_loss > total_mse_loss / len(val_loader):
            torch.save(net.state_dict(), './models/bestmodel_ae.pth')
            prev_loss = total_mse_loss / len(val_loader)
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    # pickle.dump(save_loss, open("./models/all_losses.p", "wb"))
    

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ********************************
    # Dataset stuff
    transform2apply = transforms.Compose([
                                            transforms.Resize((img_size,img_size)),
                                            transforms.ToTensor()
                                        ])
                                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    dataset = md.DatasetMario(file_path="./data", csv_name="allitems.csv", transform_in=transform2apply)

    validation_split = 0.8
    use_cuda = True

    train_size = int(validation_split * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, validation_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Train size: {}".format(len(train_ds)))
    print("Validation size: {}".format(len(validation_ds)))

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_ds, batch_size=128, shuffle=True, num_workers=4)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    CNN = FeatureExtraction(device).to(device)
    # 0.0026 mse for 1e-3 fixed
    # 64762.2282; MSE 0.0026 with 1e-3 and scheduler step_size=10, gamma=0.96
    # 59824.9376; MSE 0.0024 with 5e-3 and scheduler step_size=4, gamma=0.86
    trainNet(CNN, train_loader, validation_loader, n_epochs=35, learning_rate=1e-3, device=device)
    print("Done!")

 