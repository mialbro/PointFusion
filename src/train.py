import numpy as np
import torch
import torch.nn as nn
import torchvision
from Dataset import PointFusionDataset
import Loss
from torchvision import transforms, datasets
from PointFusion import PointFusion

def saveCheckpoint(model, epoch, optimizer, loss, path):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path)

def train(model, train_loader, n_epochs, optimizer, loss_fn):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    cnt = 0
    for epoch in range(n_epochs):
        losses = []
        train_loss = 0.0
        model.train()
        for batch_cnt, (img, cloud, offsets) in enumerate(train_loader):
            cloud = cloud.permute(0, 2, 1)
            img = img.to(device=device)
            cloud = cloud.to(device=device)
            offsets = offsets.to(device=device)
            # forward pass: predict outputs
            pred_offsets, pred_scores = model(img, cloud)
            # backward pass: compute gradient of loss wrt model parameters
            loss = loss_fn(pred_offsets, pred_scores, offsets)
            loss.backward()
            optimizer.zero_grad()
            # gradient descent
            optimizer.step()
            train_loss += loss.item()
            losses.append(loss.item())
            if batch_cnt % 500 == 0:
                cnt += 1
                saveCheckpoint(model, epoch, optimizer, loss, 'models/pointfusion_{}.pth'.format(cnt))
                print('Epoch: {}, Loss: {}'.format(epoch, loss))
    return model

def validate(model, val_loader, loss_fn):
    cnt = 0
    running_loss = 0.0
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for batch_cnt, (img, cloud, offsets) in enumerate(val_loader):
            batch_size = img.shape[0]
            # get data from loader
            cloud = cloud.permute(0, 2, 1)
            img = img.to(device=device)
            cloud = cloud.to(device=device)
            offsets = offsets.to(device=device)
            # forward pass: predict outputs
            pred_offsets, pred_scores = model(img, cloud)
            loss = loss_fn(pred_offsets, pred_scores, offsets)
            running_loss += loss
            cnt += 1
    print('Accuracy: {}'.format(running_loss / cnt))

def loadData(root_dir, batch_size, pnt_cnt):
    train_dataset = PointFusionDataset(root_dir=root_dir, mode='train', pnt_cnt=pnt_cnt)
    dataset_size = train_dataset.length
    train_size = int(0.8 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

def main():
    model = PointFusion()
    batch_size = 1
    train_loader, test_loader = loadData(root_dir='datasets/Linemod_preprocessed', batch_size=batch_size, pnt_cnt=400)
    train_loss_fn = Loss.unsupervisedLoss
    test_loss_fn = Loss.cornerLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    n_epochs = 100
    model = train(model, train_loader, n_epochs, optimizer, train_loss_fn)
    validate(model, test_loader, n_epochs, test_loss_fn)
    torch.save(model, 'models/pointfusion.pth')

if __name__ == '__main__':
    main()