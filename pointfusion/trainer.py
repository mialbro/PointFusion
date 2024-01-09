import torch
import open3d as o3d
import numpy as np

import pointfusion.loss as LOSS
from pointfusion.datasets import LINEMOD

class Trainer:
    """
    Trainer wrapper class
    Attributes:
        lr (float): Learning rate
        epochs (int): Number of times to iterate dataset
        weight_decay (float): How much to decrease weight values
        batch_size (int): Number of data in single batch
        modalities (list[pointfusion.Modality]): List of input modalities
        loss_fcn (lambda): Loss function
    """
    def __init__(self) -> None:
        # hyperparameters
        self.lr = 0.01
        self.epochs = 20
        self.weight_decay = 0.1
        self.batch_size = 10
        self.modalities = []
        self.loss_fcn = None
        self._dataset = None
        self._test_set = None
        self._train_set = None
        self._train_loader = None
        self._val_loader = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_checkpoint(self, epoch: int) -> None:
        """
        Saves current status of model at given epoch
        """
        torch.save(self._model.state_dict(), f'../weights/pointfusion_{epoch}.pt')

    @property
    def model(self) -> torch.nn.Module:
        """
        Gets pointfusion model
        """
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model.to(self._device)
        self._model.train()

    @property
    def dataset(self) -> torch.utils.data.Dataset:
        """
        Gets pointfusion dataset
        """
        return self._dataset
    
    @dataset.setter
    def dataset(self, dataset: torch.utils.data.Dataset) -> None:
        self._dataset = dataset
        self._train_set, self._test_set = dataset.split(0.8)
        self._train_loader = torch.utils.data.DataLoader(dataset=self._train_set, batch_size=self.batch_size, shuffle=True)
        self._val_loader = torch.utils.data.DataLoader(dataset=self._test_set, batch_size=self.batch_size, shuffle=True)

    def fit(self) -> None:
        """
        Runs optimization
        """
        # loss and optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self.model.train()

        stats = {'train_loss': [], 'validation_loss': [], 'epoch_loss': []}
        for epoch in range(self.epochs):
            # Training
            running_loss = 0.0
            for batch_idx, (id, image, cloud, corners) in enumerate(self._train_loader):
                # output from database
                cloud = cloud.to(self._device)
                image = image.to(self._device)
                corners = corners.to(self._device).float()

                # forward
                output = self._model(image, cloud)
                loss = self.loss_fcn(output, corners)
                print(f'EPOCH {epoch} / {self.epochs} | BATCH : {batch_idx} / {len(self._train_loader)} | LOSS : {loss}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            with torch.no_grad():
                for batch_idx, (id, image, cloud, corners) in enumerate(self._val_loader):
                    cloud = cloud.to(self._device)
                    image = image.to(self._device)
                    corners = corners.to(self._device).float()
                    output = self._model(image, cloud)
                    loss = self.loss_fcn(output, corners)
                    print(f'VALIDATION EPOCH {epoch} / {self.epochs} | BATCH : {batch_idx} / {len(self._train_loader)} | LOSS : {loss}')

            stats['epoch_loss'].append(running_loss / len(self._train_loader))
            stats['train_loss'].append(loss.item())
            print(f'EPOCH LOSS " {stats["epoch_loss"][-1]}')