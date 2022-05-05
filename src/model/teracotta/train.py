from model import *
from data import *
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

## write a dataloader for the data

##  get train and val loader
def train(net, train_loader,val_loader, criterion, optimizer, num_epochs):
    net.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            optimizer.zero_grad()
            pointsx = data['x']
            pointsy = data['y']
            embx = net(pointsx)
            emby = net(pointsy)
            loss = criterion(embx, emby)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))
        torch.save(net.state_dict(), './checkpoint/model' + epoch+'.pth')
        evaluate(net, val_loader, criterion)
def evaluate(net, val_loader, criterion, optimizer):
    net.eval()
    # for epoch in range(num_epochs):
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(val_loader, 0):
            optimizer.zero_grad()
            pointsx = data['x']
            pointsy = data['y']
            embx = net(pointsx)
            emby = net(pointsy)
            loss = criterion(embx, emby)

            running_loss += loss.item()
        print('Val loss: %.3f' % ( running_loss / len(val_loader)))
if __name__ == '__main__':
    net = PointNet(dim=256)
    train_loader = ...
    val_loader = ... 
    num_epochs = 10
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train(net, train_loader,val_loader, criterion, optimizer, num_epochs)
