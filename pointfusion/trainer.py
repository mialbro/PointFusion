import argparse
from typing import Optional, Dict, List

import torch
import numpy as np
import open3d as o3d

from pointfusion.loss import dense_fusion, global_fusion
from pointfusion.models import DenseFusion, GlobalFusion
from pointfusion.datasets import LINEMOD
from pointfusion.enums import Modality, FusionMethod

class Trainer:
    """
    Trainer wrapper class
    Args:
        lr (Optional[float]): Learning rate
        epochs (Optional[int]): Number of times to iterate dataset
        weight_decay (Optional[float]): How much to decrease weight values
        batch_size (Optional[int]): Number of data in single batch
        modality (Optional[list[pointfusion.Modality]]): List of input modalities
    Attributes:
        lr (float): Learning rate
        epochs (int): Number of times to iterate dataset
        weight_decay (float): How much to decrease weight values
        batch_size (int): Number of data in single batch
        modality (list[pointfusion.Modality]): List of input modalities
        loss_fcn (lambda): Loss function
    """
    def __init__(self,
        num_points: Optional[int],
        modality: Optional[Modality] = Modality.POINTCLOUD,
        fusion_method: Optional[FusionMethod] = FusionMethod.DENSE,
        lr: Optional[float] = 0.1,
        epochs: Optional[int] = 20,
        weight_decay: Optional[float] = 0.1,
        batch_size: Optional[int] = 10
    ) -> None:
        self.init_loss = None
        # hyperparameters
        self.lr = lr
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.weight_path = None
        self._test_set = None
        self._train_set = None
        self._train_loader = None
        self._val_loader = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Set the model and loss function
        if fusion_method is FusionMethod.DENSE:
            self.model = DenseFusion(num_points=num_points, modality=modality)
            self.loss_fcn = dense_fusion
        elif fusion_method is FusionMethod.GLOBAL:
            self.model = GlobalFusion(num_points=num_points, modality=modality)
            self.loss_fcn = global_fusion
        # Set the dataset
        self.dataset = LINEMOD(num_points=400, modality=modality, fusion_method=fusion_method)

    def print_training_results(self, stats: Dict[str, float]) -> None:
        """Print current training statistics
        Args:
            stats (Dict[str, float]): Traning statistics
        """
        print(f'Epoch : {stats["epoch"]} / {self.epochs}')
        print(f'Training Loss {stats["training_loss"]}')
        print(f'Validation Loss {stats["validation_loss"]}')

    def save_checkpoint(self, epoch: int) -> None:
        """Saves current status of model at given epoch
        Args:
            epoch (int): Current epoch
        """
        torch.save(self._model.state_dict(), f'../weights/pointfusion_{epoch}.pt')

    @property
    def model(self) -> torch.nn.Module:
        """Gets pointfusion model
        Returns:
            torch.nn.Module: Trained model
        """
        return self._model

    @model.setter
    def model(self, model: torch.nn.Module) -> None:
        """Set the trainer model
        Args:
            model (torch.nn.Module): Model to set
        """
        self._model = model.to(self._device)
        self._model.train()

    @property
    def dataset(self) -> torch.utils.data.Dataset:
        """Gets pointfusion dataset
        Returns:
            torch.utils.data.Dataset: Current dataset
        """
        return self._dataset

    @dataset.setter
    def dataset(self, dataset: torch.utils.data.Dataset) -> None:
        """Set the trainer dataset
        Args:
            dataset (torch.utils.data.Dataset)
        """
        self._dataset = dataset
        self._train_set, self._test_set = dataset.split(0.8)
        self._train_loader = torch.utils.data.DataLoader(
            dataset=self._train_set,
            batch_size=self.batch_size,
            shuffle=True
        )
        self._val_loader = torch.utils.data.DataLoader(
            dataset=self._test_set,
            batch_size=self.batch_size,
            shuffle=True
        )

    def save(self, path: Optional[str] = None) -> None:
        """Save model to file
        Args:
            path (str): Path to write to
        Returns:
            None
        """
        if path is None:
            path = self.weight_path
        torch.save(self.model.state_dict(), path)

    def fit(self) -> None:
        """Runs optimization"""
        # loss and optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        self.model.train()
        stats = [{}] * self.epochs
        for epoch in range(self.epochs):
            stats[epoch] = {'training_loss': [], 'validation_loss': [], 'epoch_loss': []}
            # Training
            running_loss = 0.0
            for batch_idx, (_, image, cloud, corners) in enumerate(self._train_loader):
                # output from database
                cloud = cloud.to(self._device)
                image = image.to(self._device)
                corners = corners.to(self._device).float()
                # forward
                output = self._model(image, cloud)
                loss = self.loss_fcn(output, corners)
                if self.init_loss is None:
                    self.init_loss = loss.item()
                # Print statistics
                txt = f'{epoch}.{batch_idx} / {self.epochs}.{len(self._train_loader)} = {loss}'
                print(txt)
                # Zero gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            with torch.no_grad():
                for batch_idx, (_, image, cloud, corners) in enumerate(self._val_loader):
                    cloud = cloud.to(self._device)
                    image = image.to(self._device)
                    corners = corners.to(self._device).float()
                    output = self._model(image, cloud)
                    loss = self.loss_fcn(output, corners)

def main() -> None:
    """Run training loop"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fusion_method',
        type=FusionMethod,
        choices=list(FusionMethod),
        default=FusionMethod.DENSE
    )
    parser.add_argument(
        '--modality',
        type=Modality,
        choices=list(Modality),
        nargs='+',
        default=Modality.POINTCLOUD
    )
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--num_points', type=int, default=400)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    args = parser.parse_args()
    # Load model
    trainer = Trainer(
        num_points=args.num_points,
        modality=args.modality,
        fusion_method=args.fusion_method,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    trainer.fit()

if __name__ == '__main__':
    main()
