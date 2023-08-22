import torch
import open3d as o3d
import numpy as np

from loss import supervised, unsupervised, corner_loss
from models import PointFusion
from datasets import LINEMOD

class Trainer:
    def __init__(self) -> None:
        # hyperparameters
        self.lr = 1e-2
        self.epochs = 20
        self.weight_decay = 0.1
        self._batch_size = 10
        self._dataset = None
        self._test_set = None
        self._train_set = None
        self._train_loader = None
        self._val_loader = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def save_checkpoint(self, epoch):
        torch.save(self._model.state_dict(), f'../weights/pointfusion_{epoch}.pt')

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, model):
        self._model = model.to(self._device)
        self._model.train()

    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, dataset):
        self._dataset = dataset
        self._train_set, self._test_set = dataset.split(0.8)
        self._train_loader = torch.utils.data.DataLoader(dataset=self._train_set, batch_size=self.batch_size, shuffle=True)
        self._val_loader = torch.utils.data.DataLoader(dataset=self._test_set, batch_size=self.batch_size, shuffle=True)

    @property
    def batch_size(self):
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    def fit(self):
        # loss and optimizer
        criterion = unsupervised
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        stats = {'train_loss': [], 'val_loss': []}
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            for batch_idx, (id, image, cloud, gt_corners, gt_offsets) in enumerate(self._train_loader):
                self._optimizer.zero_grad()
                # output from database
                cloud = cloud.to(self._device)
                image = image.to(self._device)
                gt_corners = gt_corners.to(self._device)
                gt_offsets = gt_offsets.to(self._device)

                # forward
                pred_offsets, pred_scores = self._model(image, cloud)
                loss = criterion(pred_offsets, pred_scores, gt_offsets)

                gt_corners = gt_corners.permute(0, 2, 1)

                # backward
                loss.backward()
                # gradient descent
                self._optimizer.step()

                running_loss += loss.item()
                stats['train_loss'].append(loss.item())
                #print(f'Epoch: {epoch + 1}/{self.epochs} | Batch: {batch_idx+1}/{len(self._train_loader)} | Loss: {loss.item():.4f}')
                B = cloud.size()[0]
                indices = torch.argmax(pred_scores, dim=1, keepdim=True).detach()
                score = torch.gather(pred_scores, 1, indices)
                #print((cloud.size(), indices.size(), pred_offsets.size()))
                points = torch.gather(cloud, 2, indices.view(B, 1, 1).expand(-1, 3, 1))
                pred_offset = torch.gather(pred_offsets, 3, indices.view(B, 1, 1, 1).expand(-1, 3, 8, 1))
                
                # corners
                pred_corners = torch.zeros(cloud.size()[0], 3, 8).cuda()
                pred_corners[:, :, 0] = (points - pred_offset[:, :, 0])[:, :, 0]
                pred_corners[:, :, 1] = (points - pred_offset[:, :, 1])[:, :, 0]
                pred_corners[:, :, 2] = (points - pred_offset[:, :, 2])[:, :, 0]
                pred_corners[:, :, 3] = (points - pred_offset[:, :, 3])[:, :, 0]
                pred_corners[:, :, 4] = (points - pred_offset[:, :, 4])[:, :, 0]
                pred_corners[:, :, 5] = (points - pred_offset[:, :, 5])[:, :, 0]
                pred_corners[:, :, 6] = (points - pred_offset[:, :, 6])[:, :, 0]
                pred_corners[:, :, 7] = (points - pred_offset[:, :, 7])[:, :, 0]

                #print()
                #print(f'CORNER LOSS : {(pred_corners - gt_corners).abs().mean().item()}')
                #print(f'SCORES : {score.tolist()}')

            #self.save_checkpoint(epoch)

            # Validation
            '''
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for (id, cropped_image, cloud, corners, corner_offsets) in self._val_loader:
                    image = cropped_image.to(self._device)
                    corners = corners.to(self._device)
                    corners = corners.permute(0, 2, 1)
                    targets = corner_offsets.to(self._device)
                    cloud = cloud.to(self._device)
    
                    outputs = self.model(image, cloud)  # forward pass
                    #import pdb; pdb.set_trace()
                    loss = criterion(outputs[0], outputs[1], targets)  # calculate the loss
                    val_loss += loss.item()
                    stats['val_loss'].append(val_loss)
                #print('Validation Loss: {:.4f}'.format(val_loss / len(self._val_loader)))
            '''

if __name__ == '__main__':
    trainer = Trainer()
    trainer.batch_size = 1
    trainer.model = PointFusion()
    trainer.dataset = LINEMOD(point_count=100)
    trainer.fit()