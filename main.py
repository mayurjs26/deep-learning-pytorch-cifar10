import time

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, TensorDataset

dataset = CIFAR10(root='data/', download=False, transform=ToTensor())
test_dataset = CIFAR10(root='data/', train=False, transform=ToTensor())

classes = dataset.classes
torch.manual_seed(43)
val_size = 5000
train_size = len(dataset) - val_size
input_size = 3*32*32
output_size = 10

epochs = 2 # how many epochs to train for
bs = 128
learning_rate = 1e-3

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


class CIFAR_CNN(ImageClassificationBase):
    def __init__(self):
        super().__init__()

        # without batchnorm
        # self.conv1 = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        #
        # self.conv2 = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(9, 18, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
        # self.conv5 = nn.Conv2d(18, 36, kernel_size=3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1)
        # self.conv7 = nn.Flatten()
        # self.conv8 = nn.Linear(576, 100)
        # self.conv9 = nn.Dropout()
        # self.conv10 = nn.Linear(100,10)

        #batchnorm
        self.conv1 = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        self.convbn1 = nn.BatchNorm2d(9)
        self.conv2 = nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1)
        self.convbn2 = nn.BatchNorm2d(9)
        self.conv3 = nn.Conv2d(9, 18, kernel_size=3, stride=1, padding=1)
        self.convbn3 = nn.BatchNorm2d(18)
        self.conv4 = nn.Conv2d(18, 18, kernel_size=3, stride=1, padding=1)
        self.convbn4 = nn.BatchNorm2d(18)
        self.conv5 = nn.Conv2d(18, 36, kernel_size=3, stride=1, padding=1)
        self.convbn5 = nn.BatchNorm2d(36)
        self.conv6 = nn.Conv2d(36, 36, kernel_size=3, stride=1, padding=1)
        self.convbn6 = nn.BatchNorm2d(36)
        self.conv7 = nn.Flatten()
        self.conv8 = nn.Linear(576, 100)
        self.conv9 = nn.Dropout()
        self.conv10 = nn.Linear(100, 10)



    def forward(self, xb):

        # without batchnorm
        # xb = xb.view(-1, 3, 32, 32)
        # xb = F.relu(self.conv1(xb))
        # xb = F.relu(self.conv2(xb))
        # xb = F.max_pool2d(xb, stride = 2, kernel_size = 2)
        # xb = F.relu(self.conv3(xb))
        # xb = F.relu(self.conv4(xb))
        # xb = F.max_pool2d(xb, stride = 2, kernel_size = 2)
        # xb = F.relu(self.conv5(xb))
        # xb = F.relu(self.conv6(xb))
        # xb = F.max_pool2d(xb, stride=2, kernel_size=2)
        # xb = self.conv7(xb)
        # xb = self.conv8(xb)
        # xb = F.relu(xb)
        # xb = self.conv9(xb)
        # xb = self.conv10(xb)

        #batchnorm
        xb = xb.view(-1, 3, 32, 32)
        xb = self.conv1(xb)
        xb = F.relu(self.convbn1(xb))
        xb = self.conv2(xb)
        xb = F.relu(self.convbn2(xb))
        xb = F.max_pool2d(xb, stride = 2, kernel_size = 2)
        xb = self.conv3(xb)
        xb = F.relu(self.convbn3(xb))
        xb = self.conv4(xb)
        xb = F.relu(self.convbn4(xb))
        xb = F.max_pool2d(xb, stride = 2, kernel_size = 2)
        xb = self.conv5(xb)
        xb = F.relu(self.convbn5(xb))
        xb = self.conv6(xb)
        xb = F.relu(self.convbn6(xb))
        xb = F.max_pool2d(xb, stride=2, kernel_size=2)
        xb = self.conv7(xb)
        xb = self.conv8(xb)
        xb = F.relu(xb)
        xb = self.conv9(xb)
        xb = self.conv10(xb)
        return xb.view(-1, xb.size(1))


def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.has_mps:
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    start = time.time()
    batch_size=16
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    device = get_default_device()
    print(device)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)

    model = to_device(CIFAR_CNN(), device)
    print(count_parameters(model))
    print("Validation evalution : " + str(evaluate(model, val_loader)))
    print("Training : ")
    fit(10, 0.01, model, train_loader, val_loader)
    print("Validation evalution : ")
    print(evaluate(model, test_loader))
    print("Time Taken : " + str(time.time() - start))

