# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, NonLocal2d

from .base_decoder import resize
from .base_decoder import BaseDecodeHead
from einops import rearrange

up_kwargs = {'mode': 'bilinear', 'align_corners': False}
norm_cfg = dict(type='BN', requires_grad=True)


class BSFPNHead(BaseDecodeHead):
    def __init__(self, in_channels=[256, 512, 1024, 2048], in_index=[0, 1, 2, 3], num_classes=2, channels=512):
        super().__init__(input_transform='multiple_select',
                         in_channels=in_channels, in_index=[0, 1, 2, 3], num_classes=num_classes,
                         channels=512, dropout_ratio=0.1, norm_cfg=norm_cfg, align_corners=False)
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        # FPN Module
        self.lateral_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = nn.Conv2d(in_channels, self.channels, 1 )
            self.lateral_convs.append(l_conv)
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        c1, c2, c3, c4 = inputs

        #  in_channels=[256, 512, 1024, 2048]
        # ========= output ==============
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(c4)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        out = self.cls_seg(laterals[0])
        return out
