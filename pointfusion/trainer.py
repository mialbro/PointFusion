import open3d as o3d

import torch
import numpy as np
import pointfusion

class Trainer:
    def __init__(self) -> None:
        # hyperparameters
        self.lr = 1e-3
        self.weight_decay = 1e-5
        self.batch_size = 10
        self.epochs = 20
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

    def fit(self):
        # loss and optimizer
        criterion = pointfusion.loss.unsupervised
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        stats = {'train_loss': [], 'val_loss': []}
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            running_loss = 0.0
            for batch_idx, (id, cropped_image, cloud, corners, corner_offsets) in enumerate(self._train_loader):
                self._optimizer.zero_grad()
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
                self._optimizer.step()
                running_loss += loss.item()
                stats['train_loss'].append(loss.item())
                print(f'Epoch: {epoch + 1}/{self.epochs} | Batch: {batch_idx+1}/{len(self._train_loader)} | Loss: {loss.item():.4f}')
                if batch_idx ==20:
                    break
            
            #self.save_checkpoint(epoch)

            # Validation
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
                    import pdb; pdb.set_trace()
                    '''
                    loss = criterion(outputs[0], outputs[1], targets)  # calculate the loss
                    val_loss += loss.item()
                    stats['val_loss'].append(val_loss)
                    '''
                #print('Validation Loss: {:.4f}'.format(val_loss / len(self._val_loader)))

if __name__ == '__main__':
    trainer = Trainer()
    trainer.model = pointfusion.models.PointFusion()
    trainer.dataset = pointfusion.datasets.LINEMOD()
    trainer.fit()