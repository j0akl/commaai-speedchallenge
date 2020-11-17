import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

labels = open('data/train.txt', 'r').readlines()
# for i in train_data:

dict_data = np.load('data/train_flow.npz')
raw_train_data = torch.tensor(dict_data['arr_0'])
# resize data, put channels first
raw_train_data.resize_((1000, 2, 480, 640))
train_data = []
for i in range(0, len(raw_train_data)):
    train_data.append([raw_train_data[i], float(labels[i][:-2])])

train_ds, val_ds = random_split(train_data, [900, 100])

batch_size = 50
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=batch_size*2)

# img, label = train_data[0]

def apply_kernel(image, kernel):
    ri, ci = image.shape
    rk, ck = kernel.shape
    ro, co = ri-rk+1, ci-ck+1  # output dimensions
    output = torch.zeros([ro, co])
    for i in range(ro): 
        for j in range(co):
            output[i,j] = torch.sum(image[i:i+rk,j:j+ck] * kernel)
    return output 

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds - labels < 1).item() / len(preds))

# creates the model, sequential layers defined in the next step
class SpeedBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        labels.resize_((50, 1))
        out = self(images)
        loss_fn = nn.MSELoss(reduction='mean')
        loss = loss_fn(out.float(), labels.float())
        return loss

    def validation_step(self, batch):
        images, labels = batch
        labels.resize_((100, 1))
        out = self(images)
        loss_fn = nn.MSELoss(reduction='mean')
        loss = loss_fn(out.float(), labels.float())
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'val_loss': epoch_loss, 'val_acc': epoch_acc}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['val_loss'], result['val_acc']))

# link layers
class SpeedModel(SpeedBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(614400, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, xb):
        # for layer in self.network:
#             x = layer(xb)
#             print(x.size())
        return self.network(xb)

def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(num_epochs, lr, model, train_loader, val_loader,
        opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(num_epochs):
        # training phase
        for batch in train_loader:
            loss_double = model.training_step(batch)
            # convert to float
            loss = loss_double.float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

model = SpeedModel()

initial = evaluate(model, val_dl)
print('initial: ', initial)

history = fit(5, 3e-5, model, train_dl, val_dl)
history2 = fit(5, 3e-5, model, train_dl, val_dl)
history3 = fit(5, 3e-5, model, train_dl, val_dl)
history4 = fit(5, 3e-5, model, train_dl, val_dl)
history5 = fit(5, 3e-5, model, train_dl, val_dl)
