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
        w: Optional[float] = 0.5,
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
    score = x[0]
    corner_offset = x[1]
    fcn = torch.nn.SmoothL1Loss(reduction='none')
    corner_loss = fcn(corner_offset, y).sum(dim=2).sum(dim=2)
    # as log approaches zero it grows negatively
    loss = (corner_loss * score) - (w * torch.log(score + eps))
    loss = loss.mean()
    return loss

    def view_tensor():
        pass
        '''
        img = objects[0].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", image_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            sys.exit()
        '''