import torch

def global_fusion(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Returns error between predicted corners and ground-truth corners
    Args:
        input (torch.Tensor): predicted corners
        target (torch.Tensor): Ground truth corners
    Returns:
        Loss
    """
    mse_loss = torch.nn.MSELoss()
    output = mse_loss(input, target)
    return output

def dense_fusion(input: torch.Tensor, target, w: float = 0.1, eps: float = 1e-16) -> torch.Tensor:
    """
    Returns error between predicted corners and ground-truth corners
    Args:
        input (list[torch.Tensor]): confidence scores, predicted corners
        target (torch.Tensor): Ground truth corners
        w (Optional[float]): Scale of how much to weigh high confidence scores
        eps (Optional[float]): Epsilon for torch.log
    Returns:
        Loss
    """
    scores = input[0]
    corners = input[1]
    L1 = torch.nn.SmoothL1Loss(reduction='none')
    loss = L1(corners, target).sum(dim=(1, 2))
    loss = (loss * scores) - (w * torch.log(scores + eps)) # as log approaches zero it grows negatively
    loss = loss.mean()
    return loss