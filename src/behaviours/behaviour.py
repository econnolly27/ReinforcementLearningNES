
# Implements a Neural Network for defining button behaviours ala BBRL. Thi script also implements train and test functions.
# This is a helper file and should be run from another script, i.e. mario_bx.py
# Authors: Gerardo Aragon-Camarasa, 2020

import pickle
import time
import os

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from behaviours import plot


class VisualMotion(nn.Module):
    def __init__(self, noframes):
        super(VisualMotion, self).__init__()
        self.noframes = noframes
        self.conv1 = nn.Conv2d(noframes*128, 128, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv2(x0)
        return x1


class Behaviour(nn.Module):
    def __init__(self):
        super(Behaviour, self).__init__()
        self.fc1 = nn.Linear(128*4*4, 1) # (6*4*4, 1)

    def forward(self, x):
        x2 = x.view(x.size(0), -1)
        output = self.fc1(x2)
        return output

    @staticmethod
    def loss_function(x_net, x, criterion=None):
        if criterion is None:
            #criterion = nn.BCEWithLogitsLoss(reduction='mean')#nn.CrossEntropyLoss()#nn.NLLLoss()#BCELoss()
            criterion = nn.MSELoss(reduction='mean')

        loss = criterion(x_net, x)
        return loss


def trainNetBx(net_motion, net_bx, feat_extract, train_loader, val_loader, n_epochs, lr, device, namebx):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", lr)
    print("=" * 30)
    
    #Get training data
    n_batches = len(train_loader)

    #Create our loss and optimizer functions
    params = list(net_motion.parameters()) + list(net_bx.parameters())
    # optimizer_motion = optim.Adam(net_motion.parameters(), lr=lr)
    optimizer_bx = optim.Adam(params, lr=lr)

    scheduler = StepLR(optimizer_bx, step_size=5, gamma=0.5)
    lr_step = optimizer_bx.param_groups[0]['lr']

    #Time for printing
    training_start_time = time.time()

    for param in feat_extract.parameters():
            param.requires_grad = False
    feat_extract.eval()

    prev_loss = np.inf
    stats_train = []
    stats_val = []
    #Loop for n_epochs
    for epoch in range(n_epochs):
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        losses_list = []

        for i, data in enumerate(train_loader, 0):
            # Get inputs
            inputs = data['image']
            controller = data['state'].to(device)
            controller = Variable(controller) # .view(-1)
            controller = controller.squeeze(1)

            feats = []
            for j in range(0, len(inputs)):
                image = Variable(inputs[j].to(device))
                feats.append(feat_extract.encode(image))

            features = torch.cat(tuple(feats), dim=1)

            #Set the parameter gradients to zero
            # optimizer_motion.zero_grad()
            optimizer_bx.zero_grad()
            #Forward pass, backward pass, optimize
            output_motion = net_motion(features)
            output_control = net_bx(output_motion)
            output_control = torch.flatten(output_control)
            loss = net_bx.loss_function(output_control, controller)
            loss.backward()
            # optimizer_motion.step()
            optimizer_bx.step()

            mse = nn.MSELoss(reduction='none')(output_control, controller)
            losses_list.append(mse.data.cpu().numpy().tolist())

            # Print statistics
            total_train_loss += loss.item()
            
            #Print every ith batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                flat_list = flatten_list(losses_list)
                flat_list = np.array(flat_list)
                mu = np.mean(flat_list)
                standev = np.std(flat_list)
                var = np.var(flat_list)
                stats_train.append([epoch, mu, standev, var])

                print('{:d}% - Epoch [{}/{}], Step [{}/{}], MSE mean: {:.4f}, std: {:.4f}, var: {:.4f}. Took: {:.2f}s with LR: {:.5f}'
                  .format(int(100 * (i+1) / n_batches), epoch + 1, n_epochs, i + 1, len(train_loader), mu, standev, var, time.time() - start_time, lr_step))
                
                if epoch > 0:
                    save_dir = './plots'
                    plot.plot(save_dir, '/loss train'+namebx, mu)
                    plot.flush()
                    plot.tick()
                
                #Reset running loss and time
                start_time = time.time()
                losses_list = []

        total_train_loss = total_train_loss / len(train_loader)
        # At the end of the epoch, do a pass on the validation set
        total_val_loss, valmu, valstdev, valvar = validation_and_plots(feat_extract, net_bx, net_motion, val_loader, device)
        stats_val.append([epoch, valmu, valstdev, valvar])
        if prev_loss > total_val_loss:
            torch.save(net_motion.state_dict(), './models/bestmodel_vmt_'+ namebx + str(net_motion.noframes) + 'f.pth')
            torch.save(net_bx.state_dict(), './models/bestmodel_bx_'+ namebx + str(net_motion.noframes) + 'f.pth')
            prev_loss = total_val_loss

        scheduler.step()
        lr_step = optimizer_bx.param_groups[0]['lr']

        if epoch > 0:
            print('Plot total losses for validation and training...')
            save_dir = './plots'
            plot.plot_vt(save_dir, '/total loss '+namebx, total_train_loss, total_val_loss)
            plot.flush_vt()
            plot.tick_vt()

        save_stats(namebx, stats_train, stats_val)

    print("Training finished, took {:.2f} hr".format((time.time() - training_start_time)/3600))


def validation_and_plots(feat_extract, net_bx, net_motion, val_loader, device, prefix='./plots'):
    print('===========================================')
    n_batches = len(val_loader)
    print_every = n_batches // 10
    
    total_val_loss = 0
    losses_list = []
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):

            inputs = data['image']
            target = data['state'].to(device)
            target = Variable(target)  # .view(-1)
            target = target.squeeze(1)

            # Forward pass
            # Extract features
            feats = []
            for j in range(0,len(inputs)):
                image = Variable(inputs[j].to(device))
                feats.append(feat_extract.encode(image))

            features = torch.cat(tuple(feats), dim=1)

            net_motion.eval()
            net_bx.eval()
            output_motion = net_motion(features)
            output = net_bx(output_motion)
            output = torch.flatten(output)
            val_loss_size = net_bx.loss_function(output, target)
            total_val_loss += val_loss_size.data.item()

            mse = nn.MSELoss(reduction='none')(output, target)
            losses_list.append(mse.data.cpu().numpy().tolist())

            if (i + 1) % (print_every + 1) == 0:
                print('Validation {:d}% - Step [{}/{}]'.format(int(100 * (i+1) / n_batches), i + 1, len(val_loader)))
    
    flat_list = flatten_list(losses_list)
    flat_list = np.array(flat_list)
    mu = np.mean(flat_list)
    standev = np.std(flat_list)
    var = np.var(flat_list)
    print("Validation loss mean: {:.4f} (or {:.4f}), std {:.4f}, var {:.4f}".format(total_val_loss / len(val_loader), mu, standev, var))
    print("Validation completed!!!")
    print('===========================================')
    return total_val_loss / len(val_loader), mu, standev, var


def save_stats(bxname, stats_train, stats_val):
    save_dir = os.path.abspath('./plots')
    with open(save_dir+"/"+bxname+'_stats_train.pkl', 'wb') as fp:
            pickle.dump(stats_train, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open(save_dir+"/"+bxname+'_stats_val.pkl', 'wb') as fp:
        pickle.dump(stats_val, fp, protocol=pickle.HIGHEST_PROTOCOL)


def flatten_list(losses_list):
    flat_list = [item for sublist in losses_list for item in sublist]
    return flat_list