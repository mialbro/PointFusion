import numpy as np
import torch
import torch.nn as nn
import torchvision

def unsupervisedLoss(pred_offsets, pred_scores, offsets):
    weight = 0.1
    L1 = nn.SmoothL1Loss(reduction='none')
    loss_offset = L1(pred_offsets, offsets) # [B x pnts x 8 x 3]
    loss_offset = torch.mean(loss_offset, (2, 3)) # [B x 200]
    # [B x 200]
    loss = ((loss_offset * pred_scores) - (weight * torch.log(pred_scores)))
    loss = loss.mean() # [1]
    return loss
