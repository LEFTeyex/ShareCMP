import torch.nn as nn
from mmseg.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class CPALoss(nn.Module):
    def __init__(self,
                 loss_weight: float = 1.0,
                 loss_name: str = 'loss_mse',
                 size_average=None,
                 reduce=None,
                 reduction: str = 'mean'):
        super().__init__()
        self.loss_func = nn.MSELoss(size_average, reduce, reduction)
        self.loss_weight = loss_weight
        self.loss_name = loss_name

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        loss = self.loss_func(x, y)
        return self.loss_weight * loss
