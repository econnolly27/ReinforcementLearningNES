
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
from torch.autograd import Variable, grad
import torch.nn.functional as F

import torch.optim as optim
import time
from spectral_normalization import SpectralNorm as SN
import plot
from scipy.misc import imsave


class Encoder(nn.Module):
    def __init__(self, dim, shape=(64, 64, 3)):
        super(Encoder, self).__init__()
        self.shape = shape
        self.dim = dim
        convblock = nn.Sequential(
                nn.Conv2d(3, self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(self.dim, 2 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(4 * self.dim, 8 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                nn.Conv2d(8 * self.dim, 16 * self.dim, 3, 2, padding=1),
                nn.Dropout(p=0.3),
                nn.LeakyReLU(),
                )
        self.main = convblock
        self.linear = nn.Linear(4*4*4*self.dim, self.dim)

    def forward(self, input):
        input = input.view(-1, self.shape[2], self.shape[1], self.shape[0])
        output = self.main(input)
        output = output.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output.view(-1, self.dim)


class Generator(nn.Module):
    def __init__(self, dim, shape=(64, 64, 3)):
        super(Generator, self).__init__()
        self.shape = shape
        self.dim = dim
        preprocess = nn.Sequential(
                nn.Linear(self.dim, 2* 4 * 4 * 4 * self.dim),
                nn.BatchNorm1d(2 * 4 * 4 * 4 * self.dim),
                nn.ReLU(True),
                )
        block1 = nn.Sequential(
                nn.ConvTranspose2d(8 * self.dim, 4 * self.dim, 2, stride=2),
                nn.BatchNorm2d(4 * self.dim),
                nn.ReLU(True),
                )
        block2 = nn.Sequential(
                nn.ConvTranspose2d(4 * self.dim, 2 * self.dim, 2, stride=2),
                nn.BatchNorm2d(2 * self.dim),
                nn.ReLU(True),
                )
        block3 = nn.Sequential(
                nn.ConvTranspose2d(2 * self.dim, self.dim, 2, stride=2),
                nn.BatchNorm2d(self.dim),
                nn.ReLU(True),
                )
        deconv_out = nn.ConvTranspose2d(self.dim, 3, 2, stride=2)

        self.preprocess = preprocess
        self.block1 = block1
        self.block2 = block2
        self.block3 = block3
        self.deconv_out = deconv_out
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4 * 2 * self.dim, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.deconv_out(output)
        output = self.tanh(output)
        output = output.view(-1, self.shape[2], self.shape[1], self.shape[0])
        return output


class Discriminator(nn.Module):
    def __init__(self, dim, shape=(64, 64, 3)):
        super(Discriminator, self).__init__()
        self.shape = (64, 64, 3)
        self.dim = dim

        self.conv1 = SN(nn.Conv2d(3, self.dim, 3, 1, padding=1))
        self.conv2 = SN(nn.Conv2d(self.dim, self.dim, 3, 2, padding=1))
        self.conv3 = SN(nn.Conv2d(self.dim, 2 * self.dim, 3, 1, padding=1))
        self.conv4 = SN(nn.Conv2d(2 * self.dim, 2 * self.dim, 3, 2, padding=1))
        self.conv5 = SN(nn.Conv2d(2 * self.dim, 4 * self.dim, 3, 1, padding=1))
        self.conv6 = SN(nn.Conv2d(4 * self.dim, 4 * self.dim, 3, 2, padding=1))
        self.linear = SN(nn.Linear(4*4*4*self.dim, 1))

    def forward(self, input):
        input = input.view(-1, self.shape[2], self.shape[1], self.shape[0])
        x = F.leaky_relu(self.conv1(input))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        output = x.view(-1, 4*4*4*self.dim)
        output = self.linear(output)
        return output


def calc_gradient_penalty(batch_size, model, real_data, gen_data):
    gp = 10
    datashape = model.shape
    alpha = torch.rand(batch_size, 1)
    real_data = real_data.view(batch_size, -1)
    gen_data = gen_data.view(batch_size, -1)
    alpha = alpha.expand(batch_size, real_data.nelement()//batch_size)
    alpha = alpha.contiguous().view(batch_size, -1).cuda()
    interpolates = alpha * real_data + ((1 - alpha) * gen_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = model(interpolates)
    gradients = grad(outputs=disc_interpolates, 
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),      
            create_graph=True, 
            retain_graph=True, 
            only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp
    return gradient_penalty


def generate_image(iter, model, save_path, batch_size, dim):
    batch_size = batch_size
    datashape = model.shape
    fixed_noise_128 = torch.randn(128, dim).cuda()
    noisev = Variable(fixed_noise_128, volatile=True)
    samples = model(noisev)
    samples = samples.view(-1, *(datashape[::-1]))
    samples = samples.mul(0.5).add(0.5)
    samples = samples.cpu().data.numpy()
    save_images(samples, save_path+'/samples_{}.jpg'.format(iter))

def save_images(X, save_path, use_np=False):
    # [0, 1] -> [0,255]
    plt.ion()
    if not use_np:
        if isinstance(X.flatten()[0], np.floating):
            X = (255.99*X).astype('uint8')
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1
    nh, nw = rows, int(n_samples/rows)
    if X.ndim == 2:
        s = int(np.sqrt(X.shape[1]))
        X = np.reshape(X, (X.shape[0], s, s))
    if X.ndim == 4:
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = int(n/nw)
        i = int(n%nw)
        img[j*h:j*h+h, i*w:i*w+w] = x

    plt.imshow(img, cmap='gray')
    plt.draw()
    plt.pause(0.001)

    if use_np:
        np.save(save_path, img)
    else:
        imsave(save_path, img)

def save_model(net, optim, epoch, path):
    state_dict = net.state_dict()
    torch.save({
        'epoch': epoch + 1,
        'state_dict': state_dict,
        'optimizer': optim.state_dict(),
        }, path)

def trainNet(netG, netD, netE, train_loader, val_loader, n_epochs, learning_rate, dim, device):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    optimizerD = optim.Adam(filter(lambda p: p.requires_grad, netD.parameters()), lr=learning_rate, betas=(0.0,0.9))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    optimizerE = optim.Adam(netE.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    schedulerD = optim.lr_scheduler.ExponentialLR(optimizerD, gamma=0.99)
    schedulerG = optim.lr_scheduler.ExponentialLR(optimizerG, gamma=0.99) 
    schedulerE = optim.lr_scheduler.ExponentialLR(optimizerE, gamma=0.99)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    ae_criterion = nn.MSELoss()
    one = torch.FloatTensor([1]).cuda()
    mone = (one * -1).cuda()
    iteration = 0 
    for epoch in range(n_epochs):
        
        train_loss  = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        
        for i, data in enumerate(train_loader, 0):

            """ Update AutoEncoder """
            for p in netD.parameters():
                p.requires_grad = False
            
            netG.zero_grad()
            netE.zero_grad()

            #Get inputs
            real_data_v = Variable(data['image']).to(device)
            real_data_v2 = real_data_v.view(train_loader.batch_size, -1)
            encoding = netE(real_data_v2)
            fake = netG(encoding)
            ae_loss = ae_criterion(fake, real_data_v)
            ae_loss.backward(one)
            optimizerE.step()
            optimizerG.step()

            # I'm here

            """ Update D network """
            for p in netD.parameters():  
                p.requires_grad = True 
            for i in range(5):
                real_data_v = Variable(data['image']).cuda()
                # train with real data
                netD.zero_grad()
                D_real = netD(real_data_v)
                D_real = D_real.mean()
                D_real.backward(mone)
                # train with fake data
                noise = torch.randn(train_loader.batch_size, dim).cuda()
                noisev = Variable(noise, volatile=True)
                fake = Variable(netG(noisev).data)
                inputv = fake
                D_fake = netD(inputv)
                D_fake = D_fake.mean()
                D_fake.backward(one)

                # train with gradient penalty (wrong size here when epoch finishes)
                gradient_penalty = calc_gradient_penalty(train_loader.batch_size, netD, real_data_v.data, fake.data)
                gradient_penalty.backward()

                D_cost = D_fake - D_real + gradient_penalty
                Wasserstein_D = D_real - D_fake
                optimizerD.step()

            # Update generator network (GAN)
            noise = torch.randn(train_loader.batch_size, dim).cuda()
            noisev = Variable(noise)
            fake = netG(noisev)
            G = netD(fake)
            G = G.mean()
            G.backward(mone)
            G_cost = -G
            optimizerG.step() 

            schedulerD.step()
            schedulerG.step()
            schedulerE.step()

            # Write logs and save samples 
            save_dir = './plots'
            plot.plot(save_dir, '/disc cost', D_cost.cpu().data.numpy())
            plot.plot(save_dir, '/gen cost', G_cost.cpu().data.numpy())
            plot.plot(save_dir, '/w1 distance', Wasserstein_D.cpu().data.numpy())
            plot.plot(save_dir, '/ae cost', ae_loss.data.cpu().numpy())
            
            # Calculate dev loss and generate samples every 50 iters
            # if iteration % 10 == 9:
            #     dev_disc_costs = []
            #     with torch.no_grad():
            #         for i, data in enumerate(val_loader, 0):
            #             imgs_v = Variable(data['image']).cuda()
            #             D = netD(imgs_v)
            #             _dev_disc_cost = -D.mean().cpu().data.numpy()
            #             dev_disc_costs.append(_dev_disc_cost)
            #         plot.plot(save_dir ,'/dev disc cost', np.mean(dev_disc_costs))
            #         generate_image(iteration, netG, save_dir, val_loader.batch_size, dim)
            #         # utils.generate_ae_image(iteration, netE, netG, save_dir, args, real_data_v)

            # Save logs every 10 iters 
            if (iteration < 5) or (iteration % 10 == 9):
                plot.flush()
                print(epoch)
            plot.tick()
            # if iteration % 100 == 0:
            #     save_model(netE, optimizerE, iteration,'models/E_{}'.format(iteration))
            #     save_model(netG, optimizerG, iteration,'models/G_{}'.format(iteration))
            #     save_model(netD, optimizerD, iteration,'models/D_{}'.format(iteration))
            iteration += 1




            
    #         #Print statistics
    #         train_loss  += loss.item()
            
    #         #Print every 10th batch of an epoch
    #         if (i + 1) % (print_every + 1) == 0:
    #             print('Epoch [{}/{}] {:d}%, Step [{}/{}], Loss: {:.4f}, took: {:.2f}s'
    #               .format(epoch + 1, n_epochs, int(100 * (i+1) / n_batches), i + 1, len(train_loader), train_loss  / print_every, time.time() - start_time))
    #             #Reset running loss and time
    #             train_loss = 0.0
    #             start_time = time.time()
            
    #     #At the end of the epoch, do a pass on the validation set
    #     total_val_loss = 0
    #     batch_size = val_loader.batch_size
    #     for i, data in enumerate(val_loader, 0):

    #         #Get inputs
    #         inputs = data['image'].to(device)
            
    #         #Wrap tensors in Variables
    #         inputs = Variable(inputs)
            
    #         #Forward pass
    #         recon_batch, mu, logvar = net(inputs)
    #         total_val_loss += net.loss_function(recon_batch, inputs, mu, logvar).item()

    #         if i == 0:
    #             n = min(inputs.size(0), 8)
    #             comparison = torch.cat([inputs[:n], recon_batch.view(batch_size, 3, 128, 128)[:n]])
    #             save_image(comparison.cpu(), 'results/reconstruction_' + str(epoch) + '.png', nrow=n)
            
    #     print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        
    # print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ********************************
    # Dataset stuff
    transform2apply = transforms.Compose([
                                            transforms.Resize((64,64)),
                                            transforms.ToTensor()
                                        ])
                                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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

    train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler, num_workers=4)
    validation_loader = DataLoader(dataset, batch_size=32, sampler=valid_sampler, num_workers=4)
    # test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler, num_workers=4)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    CNNG = Generator(dim=100).cuda()
    CNND = Discriminator(dim=100).cuda()
    CNNE = Encoder(dim=100).cuda()
    trainNet(CNNG, CNND, CNNE, train_loader, validation_loader, n_epochs=1, learning_rate=2e-4, dim=100, device=device)

    print("Done!")
