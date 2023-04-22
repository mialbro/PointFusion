import torch
import torch.nn as nn

def unsupervised_loss(pred_offsets, pred_scores, offsets):
    import pdb; pdb.set_trace()
    eps = 1e-16
    weight = 0.1
    L1 = nn.SmoothL1Loss(reduction='none')
    loss_offset = L1(pred_offsets, offsets) # [B x pnts x 8 x 3]
    loss_offset = torch.mean(loss_offset, (2, 3)) # [B x pnts]
    # [B x pnts]
    loss = ((loss_offset * pred_scores) - (weight * torch.log(pred_scores + eps)))
    loss = loss.mean() # [1]
    return loss

def corner_loss(pred_corners, true_corners):
    L1 = nn.SmoothL1Loss(reduction='none')
    corner_loss = L1(pred_corners, true_corners) # [B x 8 x 3]
    corner_loss = torch.mean(corner_loss, (1, 2)) # [B x 1]
    corner_loss = corner_loss.mean()
    return corner_loss