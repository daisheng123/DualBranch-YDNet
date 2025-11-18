# Copyright (c) OpenMMLab. All rights reserved.
from typing import Union

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_reduce_loss


def _expand_onehot_labels_dice(pred: torch.Tensor,
                               target: torch.Tensor) -> torch.Tensor:
    """Expand onehot labels to match the size of prediction.

    Args:
        pred (torch.Tensor): The prediction, has a shape (N, num_class, H, W).
        target (torch.Tensor): The learning label of the prediction,
            has a shape (N, H, W).

    Returns:
        torch.Tensor: The target after one-hot encoding,
            has a shape (N, num_class, H, W).
    """
    num_classes = pred.shape[1]
    one_hot_target = torch.clamp(target, min=0, max=num_classes)
    one_hot_target = torch.nn.functional.one_hot(one_hot_target,
                                                 num_classes + 1)
    one_hot_target = one_hot_target[..., :num_classes].permute(0, 3, 1, 2)
    return one_hot_target


def dice_loss(pred: torch.Tensor,
              target: torch.Tensor,
              weight: Union[torch.Tensor, None],
              eps: float = 1e-3,
              reduction: Union[str, None] = 'mean',
              naive_dice: Union[bool, None] = False,
              beta: float = 1.0,  # 新增beta参数控制FN惩罚
              avg_factor: Union[int, None] = None,
              ignore_index: Union[int, None] = 255) -> float:
    """非对称Dice Loss，严惩欠分割（FN）"""
    if ignore_index is not None:
        num_classes = pred.shape[1]
        pred = pred[:, torch.arange(num_classes) != ignore_index, :, :]
        target = target[:, torch.arange(num_classes) != ignore_index, :, :]
        assert pred.shape[1] != 0

    input = pred.flatten(1)
    target = target.flatten(1).float()

    # 计算TP、FP、FN（基于概率值，无需二值化）
    tp = (input * target).sum(1)  # [N,]
    fp = (input * (1 - target)).sum(1)
    fn = ((1 - input) * target).sum(1)

    if naive_dice:
        # 原生Dice公式（保持兼容性）
        numerator = 2 * tp + eps
        denominator = 2 * tp + fp + fn + eps
    else:
        # ========== 非对称Dice核心公式 ==========
        numerator = (1 + beta**2) * tp + eps
        denominator = (1 + beta**2) * tp + fp + beta**2 * fn + eps

    loss = 1 - (numerator / denominator)

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@MODELS.register_module()
class DiceLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 activate=True,
                 reduction='mean',
                 naive_dice=False,
                 beta=2.0,  # 新增非对称参数（默认β=2，FN惩罚权重为4）
                 loss_weight=1.0,
                 ignore_index=255,
                 eps=1e-3,
                 loss_name='loss_dice'):
        """非对称Dice Loss，参数说明：
        Args:
            beta (float): FN惩罚系数，β越大对欠分割惩罚越重（公式中β²）
        """
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.naive_dice = naive_dice
        self.beta = beta  # 新增参数
        self.loss_weight = loss_weight
        self.eps = eps
        self.activate = activate
        self.ignore_index = ignore_index
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        one_hot_target = target
        if (pred.shape != target.shape):
            one_hot_target = _expand_onehot_labels_dice(pred, target)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.reduction)
        if self.activate:
            if self.use_sigmoid:
                pred = pred.sigmoid()
            elif pred.shape[1] != 1:
                pred = pred.softmax(dim=1)

        # 传递beta参数到dice_loss函数
        loss = self.loss_weight * dice_loss(
            pred,
            one_hot_target,
            weight,
            eps=self.eps,
            reduction=reduction,
            naive_dice=self.naive_dice,
            beta=self.beta,  # 新增参数传递
            avg_factor=avg_factor,
            ignore_index=self.ignore_index)

        return loss

    @property
    def loss_name(self):
        return self._loss_name
