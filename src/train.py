import numpy as np
import torch
import torch.nn as nn
import torchvision
from Dataset import PointFusionDataset
import Loss
from torchvision import transforms, datasets
from PointFusion import PointFusion

preprocessing = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def getDatatLoaders(root_dir, batch_size, mode, split, pnt_cnt):
    dataset = PointFusionDataset(root_dir=root_dir, mode=mode, pnt_cnt=pnt_cnt, transform=preprocessing)
    test_cnt = int(dataset.length / split)
    train_cnt = int(dataset.length - test_cnt)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_cnt, test_cnt])
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train(model, train_loader, n_epochs, optimizer, loss_fn):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    for epoch in range(n_epochs):
        losses = []
        train_loss = 0.0
        model.train()
        for batch, (img, cloud, offsets) in enumerate(train_loader):
            cloud = cloud.permute(0, 2, 1)
            offsets = offsets.permute(0, 3, 1, 2)
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
            if epoch % 500 == 0:
                print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return model

def main():
    split = 2
    n_epochs = 10
    batch_size = 1
    pnt_cnt = 200
    root_dir = '../datasets/Linemod_preprocessed'
    model = PointFusion()
    train_loader, test_loader = getDatatLoaders(root_dir, batch_size, 'train', split, pnt_cnt)
    loss_fn = Loss.unsupervisedLoss
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    model = train(model, train_loader, n_epochs, optimizer, loss_fn)

if __name__ == '__main__':
    main()
