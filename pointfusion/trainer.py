import torch
import numpy as np

import models
from enums import Mode
from datasets import LINEMOD
from loss import unsupervised_loss, corner_loss

class Trainer:
    def __init__(self) -> None:
        # hyperparameters
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 3
        self.epochs = 20
        self._dataset = None
        self._test_set = None
        self._train_set = None
        self._train_loader = None
        self._val_loader = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_checkpoint(self, model, epoch, optimizer, loss, path):
        torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss': loss}, path)

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model.to(self._device)

    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset
        self._train_set, self._test_set = dataset.split(0.8)
        self._train_loader = torch.utils.data.DataLoader(dataset=self._train_set, batch_size=self.batch_size, shuffle=True)
        self._val_loader = torch.utils.data.DataLoader(dataset=self._test_set, batch_size=self.batch_size, shuffle=True)

    def fit(self):
        # loss and optimizer
        criterion = unsupervised_loss
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            for batch_idx, (id, cropped_image, cloud, corners, corner_offsets) in enumerate(self._train_loader):
                optimizer.zero_grad()
                # output from database
                cropped_image = cropped_image.to(self._device)
                corners = corners.to(self._device)
                corner_offsets = corner_offsets.to(self._device)
                cloud = cloud.to(self._device)
                # forward
                predicted_corner_offsets, predicted_scores = self._model(cropped_image, cloud)
                loss = criterion(predicted_corner_offsets, predicted_scores, corner_offsets)
                # backward
                loss.backward()
                # gradient descent
                optimizer.step()
                running_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                total_loss = 0.0
                total_correct = 0
                for (id, cropped_image, cloud, corners, corner_offsets) in self._val_loader:
                    image = cropped_image.to(self._device)
                    corners = corners.to(self._device)
                    targets = corner_offsets.to(self._device)
                    cloud = cloud.to(self._device)

                    outputs = self.model(image, cloud)  # forward pass
                    loss = criterion(outputs, targets)  # calculate the loss
                    total_loss += loss.item() * image.size(0)  # accumulate loss
                    _, predicted = outputs.max(1)
                    total_correct += predicted.eq(targets).sum().item()  # accumulate correct predictions

                # Calculate validation metrics
                val_loss = total_loss / len(self._test_set)
                val_acc = total_correct / len(self._test_set)

trainer = Trainer()
trainer.model = models.PointFusionNet()
trainer.dataset = LINEMOD()
trainer.fit()