import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from timm.models.swin_transformer_v2 import SwinTransformerV2, swinv2_tiny_window8_256

class SwinEncoder(pl.LightningModule):
    def __init__(self, d_model):
        super().__init__()
        st = swinv2_tiny_window8_256(pretrained=True).state_dict()
        st.pop("head.fc.weight")
        st.pop("head.fc.bias")
        # st["patch_embed.proj.weight"] = st["patch_embed.proj.weight"].mean(1, keepdim=True)

        self.swin = SwinTransformerV2(
            img_size=(256, 512),
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            mlp_ratio=4.0,
            qkv_bias=True,
            patch_size=4,
            window_size=8,
            in_chans=3,
            num_classes=0,

            drop_rate=0.3,
            proj_drop_rate=0.3,
            attn_drop_rate=0.3,
            drop_path_rate=0.3,
        )
        self.swin.load_state_dict(st, strict=True)
        self.swin.head = nn.Linear(768, d_model)

    def forward(self, x, mask):
        # x: [b, 1, h, w]
        features = self.swin(x)
        b, h, w, c = features.shape
        mask = torch.zeros(b, h, w, device=self.device, dtype=torch.bool)
        return features, mask


if __name__ == "__main__":
    "swinv2_tiny_window8_256"
    model = SwinEncoder(d_model=96).cuda()
    x = torch.randn(2, 1, 128, 512).cuda()
    mask = torch.ones(2, 128, 512).bool().cuda()
    out, mask = model(x, mask)
    print(out.shape)
    print(mask.shape)
