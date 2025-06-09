import math
from typing import Callable, Optional, Sequence, Type, Union

import torch.nn as nn


__all__ = [
    'ResNet',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152',
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
    ):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: Optional[nn.Module] = None,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv1d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv1d(
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
    
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        num_leads: int,
        stem_channels: int = 64,
        base_channels: int = 64,
        num_stages: int = 4,
        strides: Sequence[int] = (1, 2, 2, 2),
        dilations: Sequence[int] = (1, 1, 1, 1),
        deep_stem: bool = False,
        avg_down: bool = False,
        frozen_stages: int = -1,
        norm_layer: Callable[..., nn.Module] = nn.BatchNorm1d,
        multi_grid: Optional[Sequence[int]] = None,
        contract_dilation: bool = False,
        block: Type[Union[BasicBlock, Bottleneck]] = BasicBlock,
        stage_blocks: Sequence[int] = (2, 2, 2, 2),
        zero_init_residual: bool = False,
        out_indices: Sequence[int] = (0, 1, 2, 3),
    ):
        super(ResNet, self).__init__()

        self.zero_init_residual = zero_init_residual
        self.out_indices = out_indices

        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4, \
            "num_stages should be in [1, 4]"
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages, \
            "strides and dilations should be lists of the same length" \
            f" as num_stages, but got {len(strides)}, {len(dilations)}" \
            f" and {num_stages}"
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.norm_layer = norm_layer
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.block = block
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = self.stem_channels

        self._make_stem_layer(num_leads, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            stage_multi_grid = multi_grid if i == len(self.stage_blocks) - 1 else None
            planes = base_channels * 2 ** i
            res_layer = self._make_res_layer(
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                multi_grid=stage_multi_grid,
            )
            self.inplanes = planes * block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * base_channels * 2 ** (len(self.stage_blocks) - 1)

        self._reset_parameters()
        self._freeze_stages()

    def _make_stem_layer(
        self,
        in_channels: int,
        stem_channels: int,
    ):
        if self.deep_stem:
            self.stem = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                self.norm_layer(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                self.norm_layer(stem_channels // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                self.norm_layer(stem_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    stem_channels,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                self.norm_layer(stem_channels),
                nn.ReLU(inplace=True),
            )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def _make_res_layer(
        self,
        planes: int,
        num_blocks: int,
        stride: int = 1,
        dilation: int = 1,
        multi_grid: Optional[Sequence[int]] = None,
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = []
            conv_stride = stride
            if self.avg_down:
                conv_stride = 1
                downsample.append(
                    nn.AvgPool1d(
                        kernel_size=stride,
                        stride=stride,
                        ceil_mode=True,
                        count_include_pad=False,
                    )
                )
            downsample.extend(
                [
                    nn.Conv1d(
                        self.inplanes,
                        planes * self.block.expansion,
                        kernel_size=1,
                        stride=conv_stride,
                        bias=False,
                    ),
                    self.norm_layer(planes * self.block.expansion),
                ]
            )
            downsample = nn.Sequential(*downsample)

        layers = []
        if multi_grid is None:
            if dilation > 1 and self.contract_dilation:
                first_dilation = dilation // 2
            else:
                first_dilation = dilation
        else:
            first_dilation = multi_grid[0]
        layers.append(
            self.block(
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                dilation=first_dilation,
                downsample=downsample,
                norm_layer=self.norm_layer,
            )
        )
        inplanes = planes * self.block.expansion
        for i in range(1, num_blocks):
            layers.append(
                self.block(
                    inplanes=inplanes,
                    planes=planes,
                    stride=1,
                    dilation=dilation if multi_grid is None else multi_grid[i],
                    norm_layer=self.norm_layer,
                )
            )
        return nn.Sequential(*layers)

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


def resnet18(
    num_leads: int,
    **kwargs,
):
    model_args = dict(
        num_leads=num_leads,
        block=BasicBlock,
        stage_blocks=[2, 2, 2, 2],
        **kwargs,
    )
    return ResNet(**model_args)


def resnet34(
    num_leads: int,
    **kwargs,
):
    model_args = dict(
        num_leads=num_leads,
        block=BasicBlock,
        stage_blocks=[3, 4, 6, 3],
        **kwargs,
    )
    return ResNet(**model_args)


def resnet50(
    num_leads: int,
    **kwargs,
):
    model_args = dict(
        num_leads=num_leads,
        block=Bottleneck,
        stage_blocks=[3, 4, 6, 3],
        **kwargs,
    )
    return ResNet(**model_args)


def resnet101(
    num_leads: int,
    **kwargs,
):
    model_args = dict(
        num_leads=num_leads,
        block=Bottleneck,
        stage_blocks=[3, 4, 23, 3],
        **kwargs,
    )
    return ResNet(**model_args)


def resnet152(
    num_leads: int,
    **kwargs,
):
    model_args = dict(
        num_leads=num_leads,
        block=Bottleneck,
        stage_blocks=[3, 8, 36, 3],
        **kwargs,
    )
    return ResNet(**model_args)
