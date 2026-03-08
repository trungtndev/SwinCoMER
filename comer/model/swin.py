import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinEncoder(pl.LightningModule):
    def __init__(self, d_model):
        super().__init__()

        self.swin = SwinTransformer(
            img_size=(224, 224),
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            patch_size=4,
            window_size=7,
            in_chans=1,
            num_classes=0,

        )
        self.swin.head = nn.Sequential(
            nn.Linear(768, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x, mask):
        # x: [b, 1, h, w]
        features = self.swin(x)
        b, h, w, c = features.shape
        mask = torch.zeros(b, h, w, device=self.device, dtype=torch.bool)
        return features, mask


if __name__ == "__main__":
    model = SwinEncoder(d_model=96).cuda()
    x = torch.randn(2, 1, 224, 224).cuda()
    mask = torch.ones(2, 224, 224).bool().cuda()
    out, mask = model(x, mask)
    print(out.shape)
    print(mask.shape)
