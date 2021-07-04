import torch
import torch.nn as nn
from Dataset import PointFusionDataset
from torchvision import transforms
from PointFusion import PointFusion
import Loss

def saveCheckpoint(model, epoch, optimizer, loss, path):
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path)

def splitTrainTest(train_set, split):
    total_size = train_set.length
    train_size = int(split * total_size)
    test_size = total_size - train_size
    print(train_size)
    return [train_size, test_size]

# predata processing
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# hyperparameters
learning_rate = 1e-3
batch_size = 10
num_epochs = 5

# load data
dataset = PointFusionDataset(root_dir='datasets/Linemod_preprocessed', mode='train', pnt_cnt=400, transform=transform)
train_set, test_set = torch.utils.data.random_split(dataset, splitTrainTest(dataset, 0.8))
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointFusion().to(device)

# loss and optimizer
criterion = Loss.unsupervisedLoss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

cnt = 0
total_loss = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch_idx, (img, cloud, target) in enumerate(train_loader):
        cloud = cloud.permute(0, 2, 1)
        img = img.to(device)
        cloud = cloud.to(device)
        target = target.to(device)
        
        # forward
        pred_offset, pred_prob = model(img, cloud)
        loss = criterion(pred_offset, pred_prob, target)
        
        # backward
        optimizer.zero_grad()
        loss.backward()

        # statistics
        running_loss += loss.item()
        total_loss.append(running_loss)
        print(loss.item())
        if batch_idx % int(len(train_loader) / 4) == 0:    # print every len(dataset)/4 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 2000))
            saveCheckpoint(model, epoch, optimizer, loss, 'models/pointfusion_{}.pth'.format(batch_idx))
            cnt += 1

        # gradient descent
        optimizer.step()