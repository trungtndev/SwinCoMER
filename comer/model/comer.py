from typing import List, Optional

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch import FloatTensor, LongTensor

from comer.utils.utils import Hypothesis

from .decoder import Decoder
from .swin import SwinEncoder
from .encoder import Encoder
from ..datamodule import vocab


class CoMER(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        use_moe: bool,
        num_experts: Optional[int],
        qk_norm: bool,
        dropout: float,
        attn_dropout: float,
        dc: int,
        cross_coverage: bool,
        self_coverage: bool,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        # self.encoder = SwinEncoder(d_model=d_model)
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            use_moe=use_moe,
            num_experts=num_experts,
            qk_norm=qk_norm,
            dropout=dropout,
            attn_dropout=attn_dropout,
            dc=dc,
            cross_coverage=cross_coverage,
            self_coverage=self_coverage,
        )

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = rearrange(feature, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")

        B = feature.shape[0]
        img_start_token = torch.full((B, 1), vocab.IMG_START_IDX, device=self.device, dtype=torch.long)
        img_end_token = torch.full((B, 1), vocab.IMG_END_IDX, device=self.device, dtype=torch.long)
        img_start_emb = self.decoder.word_embed(img_start_token)
        img_end_emb = self.decoder.word_embed(img_end_token)

        img_start_mask = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        img_end_mask = torch.zeros((B, 1), dtype=torch.bool, device=self.device)

        feature = torch.cat((img_start_emb, feature, img_end_emb), dim=1)
        mask = torch.cat((img_start_mask, mask, img_end_mask), dim=1)

        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]
        mask = torch.cat((mask, mask), dim=0)


        out, l_aux = self.decoder(feature, mask, tgt)

        return out, l_aux

    def beam_search(
        self,
        img: FloatTensor,
        img_mask: LongTensor,
        beam_size: int,
        max_len: int,
        alpha: float,
        early_stopping: bool,
        temperature: float,
        **kwargs,
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = rearrange(feature, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")

        B = feature.shape[0]
        img_start_token = torch.full((B, 1), vocab.IMG_START_IDX, device=self.device, dtype=torch.long)
        img_end_token = torch.full((B, 1), vocab.IMG_END_IDX, device=self.device, dtype=torch.long)
        img_start_emb = self.decoder.word_embed(img_start_token)
        img_end_emb = self.decoder.word_embed(img_end_token)

        img_start_mask = torch.zeros((B, 1), dtype=torch.bool, device=self.device)
        img_end_mask = torch.zeros((B, 1), dtype=torch.bool, device=self.device)

        feature = torch.cat((img_start_emb, feature, img_end_emb), dim=1)
        mask = torch.cat((img_start_mask, mask, img_end_mask), dim=1)

        return self.decoder.beam_search(
            [feature], [mask], beam_size, max_len, alpha, early_stopping, temperature
        )
