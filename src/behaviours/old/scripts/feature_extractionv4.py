
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
import torchvision.models as models
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import MarioData as md
from torch.utils.data import DataLoader
from skimage import img_as_float

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim
import time
import plot

img_size = 256
BCE_reduction = 'sum' # 'mean' or 'sum'


class FeatureExtraction(nn.Module):
    def __init__(self, device):
        super(FeatureExtraction, self).__init__()
        self._device = device

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # b, 16, 128, 128
        self.maxpool1 = nn.MaxPool2d(2, stride=2) # b, 16, 64, 64
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)  # b, 8, 32, 32
        self.maxpool2 = nn.MaxPool2d(2, stride=1)  # b, 8, 31, 31

        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=0) # b, 16, 64, 64
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1) # b, 8, 128, 128
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1) # b, 3, 256, 256

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(True)
        self.elu = nn.ELU()
    
    def encode(self, x):
        out = self.elu(self.conv1(x))
        out = self.maxpool1(out)
        out = self.elu(self.conv2(out))
        out = self.maxpool2(out)
        return out

    def decode(self, x):
        out = self.elu(self.deconv1(x))
        out = self.elu(self.deconv2(out))
        out = self.elu(self.deconv3(out))
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
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        train_loss  = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        
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
            loss.backward()            
            optimizer.step()
            
            #Print statistics
            train_loss  += loss.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print('Epoch [{}/{}] {:d}%, Step [{}/{}], Loss: {:.4f}, took: {:.2f}s'
                  .format(epoch + 1, n_epochs, int(100 * (i+1) / n_batches), i + 1, len(train_loader), train_loss  / print_every, time.time() - start_time))
                save_dir = './plots'
                plot.plot(save_dir, '/loss train', train_loss  / print_every)
                plot.flush()
                plot.tick()
                #Reset running loss and time
                train_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        batch_size = val_loader.batch_size
        for i, data in enumerate(val_loader, 0):

            #Get inputs
            inputs = data['image'].to(device)
            
            #Wrap tensors in Variables
            inputs = Variable(inputs)
            
            #Forward pass
            recon_batch = net(inputs)
            total_val_loss += net.loss_function(recon_batch, inputs).item()

            if i == 0:
                n = min(inputs.size(0), 8)
                o_img = net.to_img(inputs[:n])
                recon_img = net.to_img(recon_batch.view(batch_size, 3, img_size, img_size)[:n])
                comparison = torch.cat([o_img, recon_img])
                save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


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
    dataset = md.DatasetMario(file_path="./", csv_name="allitems.csv", transform_in=transform2apply)

    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    use_cuda = True

    # Creating data indices for training, validation and test splits:
    dataset_size = len(dataset)
    n_test = int(dataset_size * 0.05)
    n_train = dataset_size - 2 * n_test

    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:(n_train + n_test)]
    test_indices = indices[(n_train + n_test):]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    print("Train size: {}".format(len(train_sampler)))
    print("Validation size: {}".format(len(valid_sampler)))
    print("Test size: {}".format(len(test_sampler)))

    train_loader = DataLoader(dataset, batch_size=128, sampler=train_sampler, num_workers=4)
    validation_loader = DataLoader(dataset, batch_size=32, sampler=valid_sampler, num_workers=4)
    # test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler, num_workers=4)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    CNN = FeatureExtraction(device).to(device)
    # print(CNN)
    trainNet(CNN, train_loader, validation_loader, n_epochs=63, learning_rate=1e-3, device=device)

    print("Done!")
 