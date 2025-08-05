# Copyright (c) VUNO Inc. All rights reserved.

from typing import Optional, Sequence

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


__all__ = [
    'VisionTransformer',
    'vit_tiny',
    'vit_small',
    'vit_base',
]


class DropPath(nn.Module):
    '''
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    '''
    def __init__(self, drop_prob: float, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob <= 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    MLP Module with GELU activation fn + dropout.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        drop_out_rate=0.,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        fp16_enabled: bool = True,
        drop_out_rate: float = 0.,
        attn_drop_out_rate: float = 0.,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == input_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_drop_out_rate)
        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=qkv_bias)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.LayerNorm(input_dim)
            self.k_norm = nn.LayerNorm(input_dim)
        self.fp16_enabled = fp16_enabled

        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, output_dim),
                nn.Dropout(drop_out_rate),
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
            qkv,
        )

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if not self.fp16_enabled:
            with torch.cuda.amp.autocast(enabled=False):
                dots = torch.matmul(q.float(), k.transpose(-1, -2).float()) * self.scale
                attn = self.attend(dots)
                attn = self.dropout(attn)
                out = torch.matmul(attn, v.float())
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        heads: int = 8,
        dim_head: int = 32,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        fp16_enabled: bool = True,
        drop_out_rate: float = 0.,
        attn_drop_out_rate: float = 0.,
        drop_path_rate: float = 0.,
        layer_scale: Optional[float] = None,
    ):
        super().__init__()
        attn = Attention(
            input_dim=input_dim,
            output_dim=output_dim,
            heads=heads,
            dim_head=dim_head,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            fp16_enabled=fp16_enabled,
            drop_out_rate=drop_out_rate,
            attn_drop_out_rate=attn_drop_out_rate,
        )
        self.attn = PreNorm(dim=input_dim, fn=attn)
        if drop_path_rate > 0:
            self.droppath1 = DropPath(drop_path_rate)
        else:
            self.droppath1 = nn.Identity()

        ff = FeedForward(
            input_dim=output_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            drop_out_rate=drop_out_rate,
        )
        self.ff = PreNorm(dim=output_dim, fn=ff)
        if drop_path_rate > 0:
            self.droppath2 = DropPath(drop_path_rate)
        else:
            self.droppath2 = nn.Identity()

        if layer_scale is not None:
            self.ls_1 = nn.Parameter(layer_scale * torch.ones(input_dim))
            self.ls_2 = nn.Parameter(layer_scale * torch.ones(input_dim))
        else:
            self.ls_1 = self.ls_2 = 1.0

    def forward(self, x):
        x = self.droppath1(self.ls_1 * self.attn(x)) + x
        x = self.droppath2(self.ls_2 * self.ff(x)) + x
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_size: int,
        num_leads: int,
        width: int = 768,
        depth: int = 12,
        mlp_dim: int = 3072,
        heads: int = 12,
        dim_head: int = 64,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        fp16_enabled: bool = True,
        drop_out_rate: float = 0.,
        attn_drop_out_rate: float = 0.,
        drop_path_rate: float = 0.,
        uniform_dpr: bool = False,
        layer_scale: Optional[float] = None,
        frozen_stages: int = -1,
        out_indices: Sequence[int] = (3, 5, 7, 11),
        final_norm: bool = False,
        output_cls_token: bool = False,
    ):
        super().__init__()
        assert seq_len % patch_size == 0, \
            'The sequence length must be divisible by the patch size.'
        self.width = width
        self.depth = depth
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.final_norm = final_norm
        self.output_cls_token = output_cls_token

        # embedding layers
        self.num_patches = seq_len // patch_size
        self.patch_dim = num_leads * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (n p) -> b n (p c)', p=patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, width),
            nn.LayerNorm(width),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, width)
        )
        self.cls_embedding = nn.Parameter(torch.randn(width))

        # transformer layers
        if uniform_dpr:
            drop_path_rate_list = [drop_path_rate] * depth
        else:
            drop_path_rate_list = [
                x.item() for x in torch.linspace(0, drop_path_rate, depth)
            ]
        for i in range(depth):
            block = TransformerBlock(
                input_dim=width,
                output_dim=width,
                hidden_dim=mlp_dim,
                heads=heads,
                dim_head=dim_head,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                fp16_enabled=fp16_enabled,
                drop_out_rate=drop_out_rate,
                attn_drop_out_rate=attn_drop_out_rate,
                drop_path_rate=drop_path_rate_list[i],
                layer_scale=layer_scale,
            )
            self.add_module(f'block{i}', block)
        self.dropout = nn.Dropout(drop_out_rate)
        self.norm = nn.LayerNorm(width)

        self.feature_dim = width

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.to_patch_embedding.eval()
            for param in self.to_patch_embedding.parameters():
                param.requires_grad = False
            self.pos_embedding.requires_grad = False
        for i in range(self.frozen_stages):
            m = getattr(self, f'block{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, series):
        x = self.to_patch_embedding(series)
        b, n, _ = x.shape
        cls_embeddings = self.cls_embedding[None, None, :].expand(b, -1, -1)

        x = torch.cat((cls_embeddings, x), dim=1)
        x = x + self.pos_embedding[:, :n + 1]

        x = self.dropout(x)
        features = []
        for i in range(self.depth):
            x = getattr(self, f'block{i}')(x)
            if i == self.depth - 1 and self.final_norm:
                x = self.norm(x)
            if i in self.out_indices:
                xp = x[:, 1:, :].permute(0, 2, 1).contiguous()
                if self.output_cls_token:
                    xp = [xp, x[:, 0]]
                features.append(xp)

        return tuple(features)

    def no_weight_decay(self) -> set:
        return {'cls_embedding', 'pos_embedding'}


def vit_tiny(
    num_leads,
    seq_len=2250,
    patch_size=75,
    **kwargs,
):
    model_args = dict(
        seq_len=seq_len,
        patch_size=patch_size,
        num_leads=num_leads,
        width=192,
        depth=12,
        heads=3,
        mlp_dim=768,
        **kwargs,
    )
    return VisionTransformer(**model_args)


def vit_small(
    num_leads,
    seq_len=2250,
    patch_size=75,
    **kwargs,
):
    model_args = dict(
        seq_len=seq_len,
        patch_size=patch_size,
        num_leads=num_leads,
        width=384,
        depth=12,
        heads=6,
        mlp_dim=1536,
        **kwargs,
    )
    return VisionTransformer(**model_args)


def vit_base(
    num_leads,
    seq_len=2250,
    patch_size=75,
    **kwargs,
):
    model_args = dict(
        seq_len=seq_len,
        patch_size=patch_size,
        num_leads=num_leads,
        width=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        **kwargs,
    )
    return VisionTransformer(**model_args)
