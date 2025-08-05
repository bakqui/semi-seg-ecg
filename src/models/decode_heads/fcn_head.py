# Copyright (c) VUNO Inc. All rights reserved.

from typing import Callable

import torch
import torch.nn as nn


class FCNHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_classes: int,
        num_convs: int,
        kernel_size: int = 3,
        concat_input: bool = True,
        dilation: int = 1,
        in_index: int = -1,
        dropout_ratio: float = 0.1,
        align_corners: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
        act_layer: Callable[..., nn.Module] = nn.ReLU,
    ):
        super(FCNHead, self).__init__()
        self.num_classes = num_classes
        self.in_index = in_index
        self.align_corners = align_corners
        assert num_convs >= 0 and dilation > 0
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        if num_convs == 0:
            assert in_channels == channels

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    bias=False,
                ),
                norm_layer(channels),
                act_layer(inplace=True),
            )
        )
        for _ in range(num_convs - 1):
            convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size=kernel_size,
                        padding=conv_padding,
                        dilation=dilation,
                        bias=False,
                    ),
                    norm_layer(channels),
                    act_layer(inplace=True),
                )
            )
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.Sequential(*convs)
        if self.concat_input:
            self.conv_cat = nn.Sequential(
                nn.Conv1d(
                    in_channels + channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=False,
                ),
                norm_layer(channels),
                act_layer(inplace=True),
            )
        self.cls_seg = nn.Conv1d(channels, num_classes, 1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout(dropout_ratio)
        else:
            self.dropout = None

    def forward(self, inputs):
        x = inputs[self.in_index]
        output = self.convs(x)
        if self.concat_input:
            output = self.conv_cat(torch.cat([x, output], dim=1))
        if self.dropout is not None:
            output = self.dropout(output)
        output = self.cls_seg(output)
        return output
