import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
import torchvision.models as models
from torchvision.utils import save_image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle

from MarioData import DatasetMarioBxv2
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim
import time
import plot
from utils import plot_confusion_matrix, plt_roc
from torch.optim.lr_scheduler import StepLR

from feature_extractionv5 import FeatureExtraction

FEAT_PATH = "./models/bestmodel_ae_128x16x16.pth"
img_size = 256
seed = 13
batch_training = 512
batch_validation = 128
validation_split = 0.8
total_epochs = 15
learning_rate = 5e-3

class Behaviour(nn.Module):
    def __init__(self):
        super(Behaviour, self).__init__()

        # self.ap1 = nn.MaxPool2d(kernel_size=2, stride=4) # 128,16,16 -> 128,8,8
        self.conv1 = nn.Conv2d(4*128, 6, kernel_size=3, stride=4, padding=1)
        # self.conv1 = nn.Conv2d(128, 32, kernel_size=3, stride=4, padding=1)
        self.fc1 = nn.Linear(6*4*4, 1)
        # self.fc2 = nn.Linear(6, 1)
        #self.do = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(6, 1)

        self.elu = nn.ELU()
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = x0.view(x0.size(0), -1)
        output = self.fc1(x1)
        # x2 = self.elu(self.fc1(x1))
        # x1 = self.bn2(x1)
        # output = self.fc2(x2)
        # output = self.fc3(x3)
        # output = self.sigmoid(output)
        # x2 = self.bn3(x2)
        # output = self.fc3(x2)

        return output, self.sigmoid(output)

    @staticmethod
    def loss_function(x_net, x, criterion=None):
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss(reduction='mean')#nn.CrossEntropyLoss()#nn.NLLLoss()#BCELoss()

        loss = criterion(x_net, x)
        return loss

def validation_and_plots(feat_extract, net, val_loader, device, prefix='./plots'):
    pred_y = list()
    test_y = list()
    probas_y = list()
    total_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):

            # #Get inputs
            # data, target = data['image'].to(device), data['state'].to(device)
            # #Wrap tensors in Variables
            # data, target = Variable(data), Variable(target)
            # target = target.squeeze(1)   

            inputs = data['image']
            target = data['state'].to(device)
            target = Variable(target)#.view(-1)
            target = target.squeeze(1)
            
            #Forward pass
            # Extract features
            feats = []
            for j in range(0,len(inputs)):
                image = Variable(inputs[j].to(device))
                feats.append(feat_extract.encode(image))

            features = torch.cat(tuple(feats), dim=1)

            net.eval()
            output, pred_probs = net(features)
            output = torch.flatten(output)
            pred_probs = torch.flatten(pred_probs)
            val_loss_size = net.loss_function(output, target)

            # probas_y.extend(output.data.cpu().numpy().tolist())
            # pred_y.extend(output.data.cpu().max(1, keepdim=True)[1].numpy().flatten().tolist())
            # test_y.extend(target.data.cpu().numpy().flatten().tolist())

            total_val_loss += val_loss_size.data.item()

            #Total number of labels
            total += target.size(0)
            
            #Obtaining predictions from max value
            # _, predicted = torch.max(output.data, 1)

            t = Variable(torch.Tensor([0.5]))  # threshold
            predicted = (pred_probs > t.to(device)).float() * 1
            predicted = torch.flatten(predicted)

            #Calculate the number of correct answers
            correct += (predicted == target).sum().item()
        
    print("Validation loss = {:.2f}; Accuracy: {:.2f}%".format(total_val_loss / len(val_loader), (correct / total) * 100))
    # confusion = confusion_matrix(pred_y, test_y)
    # plot_confusion_matrix(confusion, classes=[0,1], prefix=prefix, normalize=True, title='Confusion matrix')
    # val_loader.dataset.classes
    # plt_roc(test_y, probas_y, prefix)
    print("Validation completed!!!")
    return total_val_loss / len(val_loader), correct / total


def trainNetBx(net, feat_extract, train_loader, val_loader, n_epochs, lr, device, namebx):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", lr)
    print("=" * 30)
    
    #Get training data
    n_batches = len(train_loader)

    #Create our loss and optimizer functions
    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True)
    # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    # scheduler = StepLR(optimizer, step_size=5, gamma=0.8)

    #Time for printing
    training_start_time = time.time()

    for param in feat_extract.parameters():
            param.requires_grad = False
    feat_extract.eval()

    prev_loss = np.inf

    #Loop for n_epochs
    for epoch in range(n_epochs):
        train_loss  = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        correct = 0
        total = 0
        total_correct = 0
        total_labels = 0

        for i, data in enumerate(train_loader, 0):
            #Get inputs
            inputs = data['image']
            controller = data['state'].to(device)
            controller = Variable(controller)#.view(-1)
            controller = controller.squeeze(1)
            
            feats = []
            for j in range(0,len(inputs)):
                image = Variable(inputs[j].to(device))
                feats.append(feat_extract.encode(image))

            features = torch.cat(tuple(feats), dim=1)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            #Forward pass, backward pass, optimize
            output_control, pred_probs = net(features)
            output_control = torch.flatten(output_control)
            pred_probs = torch.flatten(pred_probs)
            loss = net.loss_function(output_control, controller)
            loss.backward()
            optimizer.step()
            
            #Print statistics
            train_loss  += loss.item()

            #Print statistics
            total_train_loss += loss.item()

            #Total number of labels
            total += controller.size(0)
            
            #Obtaining predictions from max value
            # _, predicted = torch.max(output_control.data, 1)

            t = Variable(torch.Tensor([0.5]))  # threshold
            predicted = (pred_probs > t.to(device)).float() * 1
            predicted = torch.flatten(predicted)
            
            #Calculate the number of correct answers
            correct += (predicted == controller).sum().item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print('{:d}% - Epoch [{}/{}], Step [{}/{}], Prob [{:.2f}, {:.2f}], Loss: {:.4f}, Accuracy: {:.2f}% took: {:.2f}s with LR: {:.5f}'
                  .format(int(100 * (i+1) / n_batches), epoch + 1, n_epochs, i + 1, len(train_loader), pred_probs.min().item(), pred_probs.max().item(), train_loss / (print_every + 1), (correct / total) * 100, time.time() - start_time, lr))
                
                if epoch > 0:
                    save_dir = './plots'
                    plot.plot(save_dir, '/loss train'+namebx, train_loss  / (print_every + 1))
                    plot.plot(save_dir, '/acc train'+namebx, correct / total)
                    plot.flush()
                    plot.tick()
                
                #Reset running loss and time
                train_loss = 0.0
                total_correct += correct
                total_labels += total
                correct = 0
                total = 0
                start_time = time.time()

        # scheduler.step()
        total_train_loss = total_train_loss / len(train_loader)
        #At the end of the epoch, do a pass on the validation set
        total_val_loss, val_accuracy = validation_and_plots(feat_extract, net, val_loader, device)
        # torch.save(net.state_dict(), './models/' + namebx + "_" + str(epoch) + '.pth')
        if prev_loss > total_val_loss:
            torch.save(net.state_dict(), './models/bestmodel_'+ namebx + '.pth')
            prev_loss = total_val_loss

        if epoch > -1:
            save_dir = './plots'
            plot.plot_vt(save_dir, '/total loss '+namebx, total_train_loss, total_val_loss)
            plot.plot_vt(save_dir, '/total acc '+namebx, total_correct / total_labels, val_accuracy)
            plot.flush_vt()
            plot.tick_vt()

    print("Training finished, took {:.2f} hr".format((time.time() - training_start_time)/3600))


def create_datasets(dataset, shuffle_dataset):
    # Creating data indices for training and validation splits:
    train_size = int(validation_split * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, validation_ds = torch.utils.data.random_split(dataset, [train_size, test_size])
    print("Train size: {}".format(len(train_ds)))
    print("Validation size: {}".format(len(validation_ds)))

    train_loader = DataLoader(train_ds, batch_size=batch_training, shuffle=shuffle_dataset, num_workers=4)
    validation_loader = DataLoader(validation_ds, batch_size=batch_validation, shuffle=shuffle_dataset, num_workers=4)

    return train_loader, validation_loader

if __name__ == "__main__":

    np.random.seed(seed)
    torch.manual_seed(seed)
    shuffle_dataset = False
    use_cuda = True

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # Feature extraction
    feat_extract = FeatureExtraction(device)
    feat_extract.load_state_dict(torch.load(FEAT_PATH))
    feat_extract.to(device)

    transform2apply = transforms.Compose([
                                            transforms.Resize((img_size,img_size)),
                                            transforms.ToTensor()
                                        ])
    # {'a': 0, 'b': 1, 'down': 2, 'left': 3, 'right': 4, 'select': 5, 'start': 6, 'up': 7, 'None': 8}

    # ********************************
    # Behaviour for A!
    dataset = DatasetMarioBxv2(file_path="./", csv_name="./bx_data/allitems_ALL.csv", buttontrain='a', transform_in=transform2apply)
    train_loader, validation_loader = create_datasets(dataset, shuffle_dataset)
    bxmodel = Behaviour().to(device)
    trainNetBx(bxmodel, feat_extract, train_loader, validation_loader, n_epochs=total_epochs, lr=learning_rate, device=device, namebx='bxA')
    # ********************************

    # ********************************
    # Behaviour for B!
    # dataset = DatasetMarioBx(file_path="./", csv_name="./bx_data/allitems_B.csv", buttontrain='b', transform_in=transform2apply)
    # train_loader, validation_loader = create_datasets(dataset, shuffle_dataset)
    # bxmodel = Behaviour().to(device)
    # trainNetBx(bxmodel, feat_extract, train_loader, validation_loader, n_epochs=20, learning_rate=1e-3, device=device, namebx='bxB')
    # ********************************

    # ********************************
    # Behaviour for DOWN!
    # dataset = DatasetMarioBx(file_path="./", csv_name="./bx_data/allitems_DOWN.csv", buttontrain='down', transform_in=transform2apply)
    # train_loader, validation_loader = create_datasets(dataset, shuffle_dataset)
    # bxmodel = Behaviour().to(device)
    # trainNetBx(bxmodel, feat_extract, train_loader, validation_loader, n_epochs=20, learning_rate=1e-3, device=device, namebx='bxDown')
    # ********************************

    # ********************************
    # Behaviour for LEFT!
    # dataset = DatasetMarioBx(file_path="./", csv_name="./bx_data/allitems_LEFT.csv", buttontrain='left', transform_in=transform2apply)
    # train_loader, validation_loader = create_datasets(dataset, shuffle_dataset)
    # bxmodel = Behaviour().to(device)
    # trainNetBx(bxmodel, feat_extract, train_loader, validation_loader, n_epochs=20, learning_rate=1e-3, device=device, namebx='bxLeft')
    # ********************************
    
    # ********************************
    # Behaviour for RIGHT!
    # dataset = DatasetMarioBx(file_path="./", csv_name="./bx_data/allitems_RIGHT.csv", buttontrain='right', transform_in=transform2apply)
    # train_loader, validation_loader = create_datasets(dataset, shuffle_dataset)
    # bxmodel = Behaviour().to(device)
    # trainNetBx(bxmodel, feat_extract, train_loader, validation_loader, n_epochs=20, learning_rate=1e-3, device=device, namebx='bxRight')
    # ********************************

    print("Done!")




    