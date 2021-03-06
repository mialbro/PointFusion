import numpy as np
import torch
import torch.nn as nn
import torchvision
from customDataset import PointFusionDataset

'''
model = PointFusion()
pcl = np.zeros((3,1))
rgb = np.zeros((3, 224, 224))

pcl_tensor = torch.tensor([pcl], dtype=torch.float)
rgb_tensor = torch.tensor([rgb], dtype=torch.float)

model(pcl_tensor, rgb_tensor)
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocessing = torchvision.transforms.Compose([
    transforms.RandomSizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

# Hyperparameters
learning_rate = 1e-2
loss_fn = nn.NLLLoss()
n_epochs = 100
batch_size = 32

dataset = PointFusionDataset(csv_file='pointfusion.csv', root_dir='Linemod_preprocessed',
    transforms=preprocessing)

train_set, test_set = torch.utils.data.random_split(dataset, [2000, 3000])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=False)

model = PointFusion()
model.to(device)

criterion - nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(n_epochs):
        losses = []
        loss_train = 0.0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            output = model(data)
            loss = loss_fn(output, target)

            losses.append(loss.item())
            loss_train += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train / len(train_loader)))

def evaluation_loop():
    with torch.no_grad():
        for imgs, labels in val_loader:
            batch_size = imgs.shape[0]
            outputs = model(imgs.view(batch_size, -1))
            _, predicted = torch.max(outputs, dim=1)
            total += labels.shape[0]
            correct += int((predicted == labels).sum())
        print('Accuracy: {}'.format( correct / total))