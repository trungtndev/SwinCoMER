import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer
from timm.models.swin_transformer_v2 import SwinTransformerV2

import timm


class SwinEncoder(pl.LightningModule):
    def __init__(self, d_model):
        super().__init__()
        # state_dict = timm.create_model(
        #     "swinv2_tiny_window16_256", pretrained=True,
        #     # in_chans=1,
        #     num_classes=0,
        # ).state_dict()

        self.swin = SwinTransformerV2(
            img_size=(224, 448),
            in_chans=1,
            window_size=7,
            num_classes=0,
            drop_rate=0.2,
            proj_drop_rate=0.2,
            attn_drop_rate=0.05,
            drop_path_rate=0.15,

        )
        # self.swin.load_state_dict(state_dict)

        self.swin.head = nn.Sequential(
            nn.Linear(self.swin.num_features, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x, mask):
        # x: [b, 1, h, w]
        features = self.swin(x)

        mask = mask[:, 0::4, 0::4][:, 0::2, 0::2][:, 0::2, 0::2][:, 0::2, 0::2]
        # mask = mask[:, ::32, ::32]

        return features, mask

if __name__ == "__main__":
    model = SwinEncoder(d_model=512)
    img = torch.randn(1, 1, 224, 448)
    padding_mask = torch.ones(224, 448, dtype=torch.bool)
    padding_mask[:128, :128] = 0
    padding_mask = padding_mask.unsqueeze(0)

    output, padding_mask = model(img=img, padding_mask=padding_mask)

    print(output.shape)
    print(padding_mask.shape)


