from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        decode_head: nn.Module,
        decode_head_loss: Optional[nn.Module] = None,
        auxiliary_heads: Optional[nn.ModuleList] = None,
        auxiliary_head_losses: Optional[nn.ModuleList] = None,
        use_latent_projection: bool = False,
        projection_in_dim: Optional[int] = None,
        projection_out_dim: Optional[int] = None,
    ):
        super(EncoderDecoder, self).__init__()
        self.backbone = backbone
        self.decode_head = decode_head
        self.loss_decode = decode_head_loss
        if auxiliary_heads is not None:
            self.auxiliary_heads = auxiliary_heads
        if auxiliary_head_losses is not None:
            self.loss_aux = auxiliary_head_losses

        if use_latent_projection:  # 2-layer Conv module
            self.latent_projection = nn.Sequential(
                nn.Conv1d(
                    projection_in_dim,
                    projection_out_dim,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.ReLU(),
                nn.BatchNorm1d(projection_out_dim),
                nn.Conv1d(
                    projection_out_dim,
                    projection_out_dim,
                    kernel_size=1,
                    bias=False,
                ),
            )

    @property
    def with_auxiliary_heads(self) -> bool:
        """bool: whether the segmentor has auxiliary head"""
        return hasattr(self, 'auxiliary_heads') and self.auxiliary_heads is not None

    @property
    def with_decode_head(self) -> bool:
        """bool: whether the segmentor has decode head"""
        return hasattr(self, 'decode_head') and self.decode_head is not None

    @property
    def with_projection(self) -> bool:
        """bool: whether the segmentor has projection"""
        return hasattr(self, 'latent_projection') and self.latent_projection is not None

    def no_weight_decay(self):
        rst = set()
        if hasattr(self, 'backbone'):
            # union
            rst = rst.union(self.backbone.no_weight_decay())
        if self.with_decode_head:
            if hasattr(self.decode_head, 'no_weight_decay'):
                rst = rst.union(self.decode_head.no_weight_decay())
        if self.with_auxiliary_heads:
            if hasattr(self.auxiliary_heads, 'no_weight_decay'):
                rst = rst.union(self.auxiliary_heads.no_weight_decay())
        return rst

    def forward(
        self,
        inputs: Tensor,
        labels: Optional[Tensor] = None,
        return_loss: bool = False,
        return_latent: bool = False,
    ) -> dict:
        outputs = dict()
        seq_len = inputs.size()[2]
        x = self.backbone(inputs)
        embedding = x[-1]  # (batch_size, channels, embed_len)
        if return_latent:
            if self.with_projection:
                latent = self.latent_projection(embedding)
            else:
                latent = embedding
            outputs["latent"] = F.interpolate(
                latent,
                size=seq_len,
                mode="linear",
                align_corners=self.decode_head.align_corners,
            )

        seg_logits = self.decode_head(x)
        seg_logits = F.interpolate(
            seg_logits,
            size=seq_len,
            mode="linear",
            align_corners=self.decode_head.align_corners,
        )
        outputs["seg_logits"] = seg_logits

        if return_loss:
            outputs["loss"] = self.loss_decode(seg_logits, labels)

        if self.training and self.with_auxiliary_heads:
            aux_seg_logits_list = []
            loss_aux_list = []
            for auxiliary_head, loss_aux in zip(
                self.auxiliary_heads,
                self.loss_aux,
            ):
                aux_seg_logits = auxiliary_head(x)
                aux_seg_logits = F.interpolate(
                    aux_seg_logits,
                    size=seq_len,
                    mode="linear",
                    align_corners=auxiliary_head.align_corners,
                )
                loss_aux_list.append(loss_aux(aux_seg_logits, labels))
                if return_loss:
                    loss_aux_list.append(loss_aux(aux_seg_logits, labels))
            
            outputs["aux_seg_logits"] = aux_seg_logits_list
            
            if return_loss:
                outputs["loss_aux"] = loss_aux_list

        return outputs
