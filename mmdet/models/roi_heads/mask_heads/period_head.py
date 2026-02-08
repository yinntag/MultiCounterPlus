# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList, auto_fp16, force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmcv.cnn import (bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
import numpy as np
from .tsm_transformer import ATSM
from .msa_transformer import MSA


@HEADS.register_module()
class PeriodHead(BaseModule):
    r"""Modified from Dynamic Mask Head for
    `Instances as Queries <http://arxiv.org/abs/2105.01928>`_

    Args:

        in_channels (int): Input feature channels.
            Defaults to 256.
        loss_period (dict): The config for period loss.
        loss_periodicity (dict): The config for periodicity loss.
    """

    def __init__(self,
                 in_channels=256,
                 clip_length=64,
                 loss_period=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_periodicity=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=5.0),
                 loss_tsm=dict(type='MSELoss',
                               loss_weight=5.0),
                 **kwargs):
        init_cfg = None
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super(PeriodHead, self).__init__(init_cfg)
        self.test = True
        self.fp16_enabled = False
        self.in_channels = in_channels
        self.clip_length = clip_length
        self.tsm_embeddings = ATSM(clip_length=self.clip_length, temperature=13.544)
        # self.tsm_embeddings = MSA(clip_length=self.clip_length)
        self.loss_period = build_loss(loss_period)
        self.loss_periodicity = build_loss(loss_periodicity)
        self.loss_tsm = build_loss(loss_tsm)
        self.period_fcs = nn.ModuleList()
        for _ in range(0, 1):
            self.period_fcs.append(nn.Linear(in_channels, in_channels, bias=False))
            # self.period_fcs.append(nn.BatchNorm2d(in_channels))
            self.period_fcs.append(build_activation_layer(dict(type='ReLU', inplace=False)))
        self.fc_tsm = nn.Linear(self.clip_length, in_channels)
        self.fc_period = nn.Linear(in_channels, 64)
        self.fc_periodicity = nn.Linear(in_channels, 1)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                # nn.init.xavier_normal_(p)
            nn.init.constant_(self.conv_logits.bias, 0.)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.fc_period.bias, bias_init)
        nn.init.constant_(self.fc_periodicity.bias, bias_init)

    def restore_original_order(self, rearranged_features, batch_indices_object, frames_per_object):
        restored_features = []
        start_idx = 0

        for num_objects in batch_indices_object:
            end_idx = start_idx + num_objects * frames_per_object

            batch_features = rearranged_features[start_idx:end_idx]
            start_idx = end_idx

            for frame_idx in range(frames_per_object):
                indices = batch_features[frame_idx::frames_per_object]
                restored_features.append(indices)

        restored_features = torch.cat(restored_features, dim=0)
        return restored_features

    def tsm_label(self, ori_label):
        multi_new_label = []
        for i in range(0, ori_label.shape[0] // 64):
            indices = ori_label[i::ori_label.shape[0] // 64]
            multi_new_label.append(indices)
        new_label = torch.stack(multi_new_label, dim=0).reshape(-1, 256)
        return new_label

    @auto_fp16()
    def forward(self, proposal_feat, num_object_batches):
        """Forward function of PeriodHead.
        Args:
            proposal_feat (Tensor): query feature
                (batch_size*num_proposals, feature_dimensions)

          Returns:
            period_score (Tensor): Predicted period score with shape
                (batch_size*num_proposals, 1).
        """
        if self.test:
            proposal_feat = self.tsm_label(proposal_feat)


        periods_feat, tsm_embedding = self.tsm_embeddings(proposal_feat)
        tsm_pred = tsm_embedding.squeeze(-1)
        periods_feat = periods_feat.reshape(-1, self.in_channels)
        periods_feat = self.restore_original_order(periods_feat, num_object_batches, self.clip_length)


        for period_layer in self.period_fcs:
            period_features = period_layer(periods_feat)
        period_score = self.fc_period(period_features)
        periodicity_score = self.fc_periodicity(period_features)
        return period_score, periodicity_score, tsm_pred

    @force_fp32(apply_to=('period_pred', 'periodicity_pred', 'tsm_pred'))
    def loss(self, period_pred, period_targets, periodicity_pred, periodicity_targets, tsm_pred, tsm_targets, reduction_override=None):

        num_pos_period_pred = torch.tensor(period_pred.size()[0], dtype=float).to(period_pred.device)
        avg_factor_period_pred = reduce_mean(num_pos_period_pred)

        num_pos_periodicity_pred = torch.tensor(periodicity_pred.size()[0], dtype=float).to(periodicity_pred.device)
        avg_factor_periodicity_pred = reduce_mean(num_pos_periodicity_pred)

        num_pos_tsm_pred = torch.tensor(period_pred.size()[0], dtype=float).to(period_pred.device)
        avg_factor_tsm_pred = reduce_mean(num_pos_tsm_pred)

        loss = dict()

        loss_period = self.loss_period(
            period_pred,
            period_targets,
            avg_factor=avg_factor_period_pred,
            reduction_override=reduction_override)

        # periodicity_targets = 1 - periodicity_targets
        loss_periodicity = self.loss_periodicity(
            periodicity_pred,
            periodicity_targets,
            avg_factor=avg_factor_periodicity_pred,
            reduction_override=reduction_override)

        loss_tsm = self.loss_tsm(
            tsm_pred,
            tsm_targets,
            avg_factor=avg_factor_tsm_pred,
            reduction_override=reduction_override)

        loss['loss_period'] = loss_period
        loss['loss_periodicity'] = loss_periodicity
        loss['loss_tsm'] = loss_tsm
        return loss_period, loss_periodicity, loss_tsm

    def get_targets(self, sampling_results, gt_periods, gt_periodicity, rcnn_train_cfg):
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        period_targets = torch.cat(
            [gt_period[pos_assigned_gt_ind] for (gt_period, pos_assigned_gt_ind) in zip(gt_periods, pos_assigned_gt_inds)])
        periodicity_targets = torch.cat(
            [gt_periodicity[pos_assigned_gt_ind] for (gt_periodicity, pos_assigned_gt_ind) in zip(gt_periodicity, pos_assigned_gt_inds)])

        return period_targets, periodicity_targets
