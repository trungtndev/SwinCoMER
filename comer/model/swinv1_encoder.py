from typing import Tuple, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from timm.layers import drop_path

from .swinv1 import SwinTransformer
from torch import FloatTensor, LongTensor, BoolTensor, nn
from einops import rearrange


class SwinEncoder(pl.LightningModule):
    def __init__(self, d_model: int):
        super().__init__()
        self.swin = SwinTransformer(
            img_size=(224, 672),
            in_chans=1,
            embed_dim=12,
            patch_size=4,
            window_size=7,

            depths=[2,2,4,2],
            num_heads=[3,6,12,24],
            drop_path_rate=0.1,
            drop_rate=0.1,
            drop_path=0.1,
            drop_path2=0.1,
            attn_drop_rate=0.1,

            mlp_ratio=1,
        )
        self.linear = nn.Linear(self.swin.num_features, d_model)
    def forward(
        self, img: FloatTensor, padding_mask: Optional[BoolTensor] = None
    ) -> Tuple[FloatTensor, Optional[BoolTensor] | None]:
        img_feature, padding_mask = self.swin(x=img, padding_mask=padding_mask)
        img_feature = self.linear(img_feature)
        img_feature = rearrange(
            img_feature, "b (h w) d -> b h w d", h=self.swin.patches_resolution[0] // (2 ** 3)
        )
        padding_mask = rearrange(
            padding_mask, "b (h w) -> b h w", h=self.swin.patches_resolution[0] // (2 ** 3)
        )


        return img_feature, padding_mask

if __name__ == "__main__":
    model = SwinEncoder(d_model=512)
    img = torch.randn(1, 1, 224, 448)
    padding_mask = torch.ones(224, 448, dtype=torch.bool)
    padding_mask[:128, :128] = 0
    padding_mask = padding_mask.unsqueeze(0)

    output, mask = model(img=img, padding_mask=padding_mask)

    print(output.shape)
    print(mask.shape)



