from __future__ import print_function
import numpy as np
import torch
import torchvision
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from math import floor, ceil

"""
HOW TO IMPROVE:
    - Make more intentional choices about the structure of the network
     - maybe use a multi-layer fully connected perceptron instead of two random
     linear layers
    - use FEWER convolutional layers, the current one is too complex
    - add noise to the image (?) may help with overfit, seems to be the problem
    in the previous iteration
    - look into methods of image normalization to make the test set resemble
    the train set
    - meter the learning rate based on the epoch
    - make sure the loss is working correctly (hard to tell whats actually
    happening rn)
"""

def create_dataloaders(filepath, batch_size=32):

    labels = open("data/train.txt", "r").readlines()
    # use following in notebook
    # labels = open(parent_dir + "train.txt", "r").readlines()

    raw_data = np.load(filepath)['arr_0']

    len_data = len(raw_data)
    train_length = floor(len_data * .8)
    val_length = ceil(len_data * .2)

    n_x = np.array(raw_data)
    print(n_x.shape)
    n_y = np.array(labels).astype("float64")

    x_train = torch.from_numpy(n_x).float().permute(0, -1, 1, 2)
    print(x_train.shape)
    y_train = torch.from_numpy(n_y).float().view(-1, 1)
    
    train_data = []
    
    for i in range(len(x_train)):
        train_data.append([x_train[i], y_train[i]])
    
    train_ds, val_ds = random_split(train_data, [train_length, val_length])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

    return train_dl, val_dl

def train(model, train_loader, optimizer, epoch, device="cpu"):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, device='cpu'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target).item()

    # test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

def predict(x):
    # outputs an array of predictions
    predictions = []
    for i in range(len(x)):
        predictions.append(model(x[i].view(1, 2, 120, 160)).item())
    return predictions

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # first part
        self.conv1 = nn.Conv2d(2, 64, 2)
        self.conv2 = nn.Conv2d(64, 64, 2)
        self.conv3 = nn.Conv2d(64, 64, 2)
        self.conv4 = nn.Conv2d(64, 32, 2)

        # second
        self.conv5 = nn.Conv2d(32, 32, 2)
        self.conv6 = nn.Conv2d(32, 32, 2)
        self.conv7 = nn.Conv2d(32, 32, 2)
        self.conv8 = nn.Conv2d(32, 16, 2)

        # third
        self.conv9  = nn.Conv2d(16, 16, 2)
        self.conv10 = nn.Conv2d(16, 16, 2)
        self.conv11 = nn.Conv2d(16, 16, 2)
        self.conv12 = nn.Conv2d(16, 16, 2)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2816, 128)
        self.fc2 = nn.Linear(128, 1)
        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        # x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        # x = self.flat(x)
        return x

if __name__ == "__main__":
    device = "cpu"

    flow_path = "data/flows/train_flow.npz"
    train_dl, val_dl = create_dataloaders(flow_path, batch_size=128)

    # use the following in notebook
    # train_dl, val_dl = create_dataloaders(parent_dir + "flows/train_flow.npz", batch_size=512)

    model = Net().float().to(device)

    optimizer = optim.Adam(model.parameters())

    num_epochs = 10

    for epoch in range(num_epochs):
        train(model, train_dl, optimizer, epoch, device)
        test(model, val_dl, device)

    # f = open(parent_dir + 'models/nd_v4.pt', 'w')
    # torch.save(model.state_dict(), parent_dir + 'models/nd_v4.pt')
    # f.close()
