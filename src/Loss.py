import numpy as np
import torch
import torch.nn as nn
import torchvision

def unsupervisedLoss(pred_offsets, pred_scores, offsets):
    ''' TODO '''
    x = torch.ones(1, requires_grad=True)
    y = x + 2
    loss = y * y * 2
    return loss
