import torch
import torch.nn as nn

def supervised():
    pass

def unsupervised(pred_offsets, pred_scores, offsets, eps=1e-16, weight=0.1):
    L1 = nn.SmoothL1Loss(reduction='none')
    #import pdb; pdb.set_trace()
    loss = L1(pred_offsets, offsets).sum(dim=(1, 2))
    #loss = torch.mean(loss_offset, dim=(1, 2)) # [B x pnts]
    loss = ((loss * pred_scores) - (weight * torch.log(pred_scores + eps)))
    return loss.mean()

def corner_loss(pred_corners, true_corners):
    L1 = nn.SmoothL1Loss(reduction='none')
    corner_loss = L1(pred_corners, true_corners) # [B x 8 x 3]
    corner_loss = torch.mean(corner_loss, (1, 2)) # [B x 1]
    corner_loss = corner_loss.mean()
    return corner_loss