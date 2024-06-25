from typing import Optional, Tuple

import torch

def global_fusion(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gets error between predicted corners and ground-truth corners
    Args:
        x (torch.Tensor): predicted corners
        y (torch.Tensor): Ground truth corners
    Returns:
        Loss
    """
    mse_loss = torch.nn.MSELoss()
    return mse_loss(x, y)

def dense_fusion(
        x: Tuple[torch.Tensor],
        y: torch.Tensor,
        w: Optional[float] = 0.1,
        eps: Optional[float] = 1e-16
    ) -> torch.Tensor:
    """Gets error between predicted corners and ground-truth corners
    Args:
        x (Tuple[torch.Tensor]): confidence scores, predicted corners
        y (torch.Tensor): Ground truth corners
        w (Optional[float]): Scale of how much to weigh high confidence scores
        eps (Optional[float]): Epsilon for torch.log
    Returns:
        Loss
    """
    import pdb; pdb.set_trace()
    scores = x[0]
    corners = x[1]
    L1 = torch.nn.SmoothL1Loss(reduction='none')
    loss = L1(corners, y).sum(dim=(1, 2))
    #loss = (loss * scores) - (w * torch.log(scores + eps)) # as log approaches zero it grows negatively
    loss = loss.mean()
    #import pdb; pdb.set_trace()
    print(f'LOSS : {loss}')
    #print(corners)
    #print(y)
    #print()
    return loss