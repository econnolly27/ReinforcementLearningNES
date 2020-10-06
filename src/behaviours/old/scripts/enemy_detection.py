
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import utils
import matplotlib.pyplot as plt

import MarioData as md
from torch.utils.data import DataLoader
from skimage import img_as_float

from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import torch.nn.functional as F

import torch.optim as optim
import time


class MyCNN(torch.nn.Module):
    # Input is (3, 128, 128)

    def __init__(self):
        super(MyCNN, self).__init__()

        # Input ch = 3, output ch = 18
        self.conv1 = torch.nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=8, padding=0)

        # 4608 input features, 64 output
        self.fc1 = torch.nn.Linear(18*16*16, 64)

        # 64 input, 10 output
        self.fc2 = torch.nn.Linear(64,2)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (3, 128, 128) to (18, 128, 128)
        x = F.relu(self.conv1(x))

        # Size changes from (18, 128, 128) to (18, 16, 16)
        x = self.pool(x)

        # Reshape data to input to the NN
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the input
        x = x.view(-1, 18 * 16 *16)

        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the 2nd fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, output size)
        x = self.fc2(x)

        return x

    @staticmethod
    def createLossAndOptimizer(net, learning_rate=0.001):
        #Loss function
        loss = torch.nn.CrossEntropyLoss()
        #Optimizer
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
        return(loss, optimizer)


def trainNet(net, train_loader, val_loader, n_epochs, learning_rate, device):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = net.createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs, labels = data['image'].to(device), data['enemy'].to(device)
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            labels = labels.squeeze(1)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()

            #Total number of labels
            total = labels.size(0)
            
            #Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)
            
            #Calculate the number of correct answers
            correct = (predicted == labels).sum().item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                # print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                #         epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))

                print('Epoch [{}/{}] {:d}%, Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}% took: {:.2f}s'
                  .format(epoch + 1, n_epochs, int(100 * (i+1) / n_batches), i + 1, len(train_loader), running_loss / print_every, (correct / total) * 100, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        correct = 0
        total = 0
        for i, data in enumerate(val_loader, 0):

            #Get inputs
            inputs, labels = data['image'].to(device), data['enemy'].to(device)
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            labels = labels.squeeze(1)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()

            #Total number of labels
            total += labels.size(0)
            
            #Obtaining predictions from max value
            _, predicted = torch.max(val_outputs.data, 1)
            #Calculate the number of correct answers
            correct += (predicted == labels).sum().item()
            
        print("Validation loss = {:.2f}; Accuracy: {:.2f}%".format(total_val_loss / len(val_loader), (correct / total) * 100))
        
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


def show_data_batch(sample_batched, predictions):
    """Show image with Mario annotated for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2

    grid = utils.make_grid(images_batch)
    plt.imshow(img_as_float(grid.numpy().transpose((1, 2, 0))))

    print("Predictions: {}".format(predictions))

    plt.title(predictions)


def testNet(net, test_loader, device):
    with torch.no_grad():
        correct = 0
        total = 0
        for i, data in enumerate(test_loader, 0):
            
            inputs, labels = data['image'].to(device), data['enemy'].to(device)
            
            inputs = Variable(inputs)
            labels = Variable(labels)
            labels.squeeze(1)
            
            outputs = net(inputs)

            #Total number of labels
            total += labels.size(0)
            
            #Obtaining predictions from max value
            _, predicted = torch.max(outputs.data, 1)
            #Calculate the number of correct answers
            # correct += (predicted == labels).sum().item() # For some reason it doesn't work!
            pred = predicted.cpu().numpy()
            lab = labels.cpu().numpy()
            correct += (pred == np.transpose(lab)).sum()
            show_data_batch(data, pred)
            plt.axis('off')
            plt.ioff()
            plt.show()

        print('Test accuracy of the model: {:.2f} %'.format(100 * correct / total))


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ********************************
    # Dataset stuff
    transform2apply = transforms.Compose([
                                            transforms.Resize((128,128)),
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

    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler, num_workers=4)
    validation_loader = DataLoader(dataset, batch_size=128, sampler=valid_sampler, num_workers=4)
    test_loader = DataLoader(dataset, batch_size=4, sampler=test_sampler, num_workers=4)

    print("CUDA Available: ",torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    CNN = MyCNN().to(device)
    trainNet(CNN, train_loader, validation_loader, n_epochs=1, learning_rate=0.001, device=device)

    testNet(CNN, test_loader, device=device)

    print("Done!")
