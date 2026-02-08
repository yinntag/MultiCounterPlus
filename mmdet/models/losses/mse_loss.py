# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
import torch
from ..builder import LOSSES
from .utils import weighted_loss


def tsm_label(ori_label):
    multi_new_label = []
    for i in range(0, ori_label.shape[0] // 64):
        indices = ori_label[i::ori_label.shape[0] // 64]
        multi_new_label.append(indices)
    new_label = torch.stack(multi_new_label, dim=0).reshape(-1)
    return new_label


@weighted_loss
def mse_loss(pred, target):
    # print(pred.shape, target.shape)
    # pred = pred.squeeze(1)
    # target = tsm_label(target)
    # print('tgt_tsm:', target.shape)
    # print('pred_tsm:', pred.shape)
    # print('-' * 100)

    pred = pred.float().cuda()
    target = target.float().cuda()
    mask = (target == 0.0).logical_not().float()
    # print(f"target dtype: {target.shape}")
    # print(f"pred dtype: {pred.shape}")
    # print(f"mask dtype: {mask.shape}")
    loss = F.mse_loss(pred * mask, target, reduction='sum')
    loss = loss.cpu()
    # print(loss.shape)

    """Warpper of mse loss."""
    return loss


@LOSSES.register_module()
class MSELoss(nn.Module):
    """MSELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * mse_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
