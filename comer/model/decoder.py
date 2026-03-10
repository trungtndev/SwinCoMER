from typing import List, Optional
import copy

import torch
import torch.nn as nn
from einops import rearrange
from torch import FloatTensor, LongTensor

from comer.datamodule import vocab, vocab_size
from comer.model.pos_enc import WordPosEnc
import torch.nn.functional as F
from torch import Tensor
from functools import partial
# from fairscale.nn.moe import (
#     Top2Gate,
#     MOELayer,
# )
import torch.distributed as dist

from comer.model.module.attention import MultiheadAttention
from comer.model.module.rope import precompute_freqs_cis
from comer.model.module.arm import AttentionRefinementModule
from comer.model.module.moe import MOELayer, BaseMOELayer
from comer.model.module.top2gate import Top2Gate

from comer.utils.generation_utils import DecodeModel
import warnings


class SwiGLU(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(d_model, dim_feedforward)
        self.fc3 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.act(x_fc1) * x_fc2
        x = self.dropout(x)
        return self.fc3(x)


class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class MoE(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, num_experts):
        super(MoE, self).__init__()
        if dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        assert num_experts % world_size == 0, \
            "num_experts must be divisible by world_size"
        num_local_experts = num_experts // world_size

        warnings.warn(f"MoE world size: {world_size}")
        if num_local_experts == num_experts:
            warnings.warn(f"Using MoE with {num_experts} experts on a single device.")

        self.moe = BaseMOELayer(
            Top2Gate(model_dim=d_model, num_experts=num_experts),
            nn.ModuleList([
                FFN(d_model, dim_feedforward, dropout)
                for _ in range(num_local_experts)
            ])
        )

    def forward(self, x):
        # print("MoE input shape:", x.shape)
        # l_seq, b_size, d_model = x.shape
        # x = rearrange(x, "l b d -> (b l) 1 d")
        x = rearrange(x, "l b d -> b l d")
        x = self.moe(x)
        x = rearrange(x, "b l d -> l b d")
        # x = rearrange(x, "(b l) 1 d -> l b d", b=b_size, l=l_seq)
        return x, self.moe.l_aux


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, attn_dropout, qk_norm, use_moe, num_experts=None):
        super(TransformerDecoderLayer, self).__init__()
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=attn_dropout, qk_norm=qk_norm)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.use_moe = use_moe
        if use_moe:
            self.ffn = MoE(d_model, dim_feedforward, dropout, num_experts=num_experts)
        else:
            self.ffn = FFN(d_model, dim_feedforward, dropout)

    def forward(
            self,
            tgt: Tensor,
            tgt_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            freqs_cis: Optional[Tensor] = None,
    ) -> Tensor:
        tgt_norm = self.norm1(tgt)  # pre-norm
        tgt2, attn = self.self_attn(
            tgt_norm, tgt_norm, tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask,
            # freqs_cis=freqs_cis
        )
        tgt = tgt + self.dropout1(tgt2)

        tgt_norm = self.norm2(tgt)  # pre-norm
        if self.use_moe:
            tgt2, l_aux = self.ffn(tgt_norm)
        else:
            tgt2 = self.ffn(tgt_norm)
            l_aux = None

        tgt = tgt + self.dropout2(tgt2)
        return tgt, attn, l_aux

    # def forward(
    #         self,
    #         tgt: Tensor,
    #         tgt_mask: Optional[Tensor] = None,
    #         tgt_key_padding_mask: Optional[Tensor] = None,
    #         freqs_cis: Optional[Tensor] = None,
    # ) -> Tensor:
    #     tgt2, attn = self.self_attn(
    #         tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask,
    #         freqs_cis=freqs_cis
    #     )
    #     tgt = tgt + self.dropout1(tgt2)
    #     tgt = self.norm1(tgt)  # post-norm
    #     if self.use_moe:
    #         tgt2, l_aux = self.ffn(tgt)
    #     else:
    #         tgt2 = self.ffn(tgt)
    #         l_aux = None
    #     tgt = tgt + self.dropout2(tgt2)
    #     tgt = self.norm2(tgt)  # post-norm
    #     return tgt, attn, l_aux


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            attn_dropout: float,
            num_layers: int,
            use_moe: bool,
            num_experts: Optional[int],
            qk_norm: bool,
            end: int,
            theta: float,
    ):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.use_moe = use_moe
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                use_moe=use_moe,
                dropout=dropout,
                attn_dropout=attn_dropout,
                num_experts=num_experts,
                qk_norm=qk_norm,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)


        # self.freqs_cis = precompute_freqs_cis(dim=d_model // nhead, end=end, theta=theta)
        # self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
            self,
            tgt: Tensor,
            tgt_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        # current_freqs_cis = self.freqs_cis[:tgt.size(0)]
        current_freqs_cis = None
        l_aux = 0.0 if self.use_moe else None
        for i, mod in enumerate(self.layers):
            output, attn, cur_l_aux = mod(
                tgt=output,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                freqs_cis=current_freqs_cis,
            )

            if self.use_moe and cur_l_aux is not None:
                l_aux += cur_l_aux

        output = self.norm(output)

        return output, l_aux


class Decoder(DecodeModel):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout: float,
            attn_dropout: float,
            dc: int,
            use_moe: bool,
            num_experts: Optional[int],
            qk_norm: bool,
            cross_coverage: bool,
            self_coverage: bool,
            end: int = 512,
            theta: float = 10000.0,
    ):
        super().__init__()

        self.word_embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = WordPosEnc(d_model=d_model)

        self.model = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            attn_dropout=attn_dropout,
            num_layers=num_decoder_layers,
            use_moe=use_moe,
            num_experts=num_experts,
            qk_norm=qk_norm,
            end=end,
            theta=theta,
        )

        self.proj = nn.Linear(d_model, vocab_size)

    # def _build_attention_mask(self, n_img, n_txt):
    #     total = n_img + n_txt
    #     mask = torch.zeros(total, total, dtype=torch.bool, device=self.device)
    #
    #     text_start = n_img
    #     text_mask = torch.triu(
    #         torch.ones(n_txt, n_txt, dtype=torch.bool, device=self.device), 1
    #     )
    #
    #     mask[text_start:, text_start:] = text_mask
    #     return mask
    def _build_attention_mask(self, length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.full(
            (length, length), fill_value=1, dtype=torch.bool, device=self.device
        )
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(
            self, src: FloatTensor, src_mask: LongTensor, tgt: LongTensor
    ) -> FloatTensor:

        b, l = tgt.size()
        _, N, D = src.shape

        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt = self.word_embed(tgt)
        tgt = self.pos_enc(tgt)

        tgt = torch.cat([src, tgt], dim=1)
        tgt_mask = self._build_attention_mask(N+l)
        tgt_pad_mask = torch.cat([src_mask, tgt_pad_mask], dim=1)

        tgt = rearrange(tgt, "b l d -> l b d")

        out, l_aux = self.model(
            tgt=tgt,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )

        out = rearrange(out, "l b d -> b l d")

        out = self.proj(out)
        out = out[:, N:, :]

        return out, l_aux

    def transform(
            self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        word_out, _ = self(src[0], src_mask[0], input_ids)
        return word_out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Decoder(
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        dc=64,
        cross_coverage=True,
        self_coverage=True,
    )
    src = torch.randn(2, 16, 16, 512).to(device)
    src_mask = torch.zeros(2, 16, 16).bool().to(device)
    tgt = torch.randint(0, vocab_size, (2, 20)).to(device)
    model = model.to(device)
    model.eval()
    model_output = model(src, src_mask, tgt)
    print(model)
