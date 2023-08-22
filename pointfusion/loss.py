import torch
import torch.nn as nn
import torch.nn.functional as F

def supervised():
    pass
'''
def unsupervised(pred_offsets, pred_scores, offsets, eps=1e-16, weight=0.1):
    L1 = nn.SmoothL1Loss(reduction='none')
    #import pdb; pdb.set_trace()
    #pred_offsets = pred_offsets.
    loss = L1(pred_offsets, offsets).mean(dim=(1, 2))
    #loss = torch.abs(pred_offsets - offsets).sum(dim=1).mean(dim=1)
    loss = (loss * pred_scores).mean(dim=1)  - (weight * torch.log(pred_scores + eps)).mean(dim=1)
    #import pdb; pdb.set_trace()
    #print(f'LOSS 1 : {(loss * pred_scores)}')
    #print(f'SCORES : {pred_scores.max(dim=1)[0]}')
    #import pdb; pdb.set_trace()
    return loss.mean()
'''
def unsupervised(offset_preds, confidences, targets, eps=1e-16, weight=10.0):
    B, N = confidences.size()
    # Compute corner offset regression loss
    L1 = nn.SmoothL1Loss(reduction='none')
    offset_loss = L1(offset_preds, targets)
    
    # Compute the logarithmic penalty term
    penalty_term = weight * torch.log(confidences + eps)
    
    # Compute the final loss
    loss = torch.mean(offset_loss * confidences.view(B, 1, 1, N) - penalty_term.view(B, 1, 1, N))
    return loss

def corner_loss(pred_corners, true_corners):
    L1 = nn.SmoothL1Loss(reduction='none')
    corner_loss = L1(pred_corners, true_corners) # [B x 8 x 3]
    corner_loss = torch.mean(corner_loss, (1, 2)) # [B x 1]
    corner_loss = corner_loss.mean()
    return corner_loss