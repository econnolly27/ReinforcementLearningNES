
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

img_size = 256
hidden_size = 1024
intermediate_size = 4096
BCE_reduction = 'sum' # 'mean' or 'sum'


class VAE(nn.Module):
    def __init__(self, device, image_channels=3):
        super(VAE, self).__init__()
        self._device = device

        # Encoder
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(3, 32, kernel_size=2, stride=4, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, intermediate_size)

        # Latent space
        self.fc21 = nn.Linear(intermediate_size, hidden_size)
        self.fc22 = nn.Linear(intermediate_size, hidden_size)

        # Decoder
        self.fc3 = nn.Linear(hidden_size, intermediate_size)
        self.fc4 = nn.Linear(intermediate_size, 32 * 16 * 16)
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4, padding=0)
        self.conv5 = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)
            return z
        else:
            return mu
    
    def encode(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = out.view(out.size(0), -1)
        h1 = self.relu(self.fc1(out))
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        out = self.relu(self.fc4(h3))
        out = out.view(out.size(0), 32, 16, 16)
        out = self.relu(self.deconv1(out))
        out = self.relu(self.deconv2(out))
        out = self.relu(self.deconv3(out))
        out = self.sigmoid(self.conv5(out))
        return out

    # def bottleneck(self, h):
    #     mu, logvar = self.fc21(h), self.fc22(h)
    #     z = self.reparameterize(mu, logvar)
    #     return z, mu, logvar
        
    # def representation(self, x):
    #     return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @staticmethod
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x, reduction=BCE_reduction)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        loss = BCE + KLD

        # loss = nn.MSELoss()(recon_x, x)

        return loss


def trainNet(net, train_loader, val_loader, n_epochs, learning_rate, device):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
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
            recon_batch, mu, logvar = net(inputs)
            loss = net.loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()            
            optimizer.step()
            
            #Print statistics
            train_loss  += loss.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print('Epoch [{}/{}] {:d}%, Step [{}/{}], Loss: {:.4f}, took: {:.2f}s'
                  .format(epoch + 1, n_epochs, int(100 * (i+1) / n_batches), i + 1, len(train_loader), train_loss  / print_every, time.time() - start_time))
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
            recon_batch, mu, logvar = net(inputs)
            total_val_loss += net.loss_function(recon_batch, inputs, mu, logvar).item()

            if i == 0:
                # check this: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/08-AutoEncoder/conv_autoencoder.py
                n = min(inputs.size(0), 8)
                comparison = torch.cat([inputs[:n], recon_batch.view(batch_size, 3, img_size, img_size)[:n]])
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

    CNN = VAE(device).to(device)
    # print(CNN)
    trainNet(CNN, train_loader, validation_loader, n_epochs=500, learning_rate=1e-3, device=device)

    print("Done!")
 