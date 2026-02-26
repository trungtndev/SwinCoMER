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
from fairscale.nn.moe import Top2Gate, MOELayer

from comer.model.transformer.attention import MultiheadAttention, precompute_freqs_cis
from comer.model.transformer.arm import AttentionRefinementModule

from comer.utils.generation_utils import DecodeModel


class SwiGLU(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(SwiGLU, self).__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(d_model, dim_feedforward)
        self.fc3 = nn.Linear(dim_feedforward, d_model)
        self.act = nn.SiLU()
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
        self.act = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.act(self.linear1(x))))


class MoE(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, num_experts):
        super(MoE, self).__init__()
        self.moe = MOELayer(
            Top2Gate(model_dim=d_model, num_experts=num_experts),
            nn.ModuleList([
                copy.deepcopy(
                    FFN(d_model, dim_feedforward, dropout)
                )
                for _ in range(num_experts)
            ])
        )

    def forward(self, x):
        return self.moe(x), self.moe.l_aux


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.ffn = MoE(d_model, dim_feedforward, dropout, num_experts=4)

    # def forward(
    #         self,
    #         tgt: Tensor,
    #         memory: Tensor,
    #         arm: Optional[AttentionRefinementModule],
    #         freqs_cis: Tensor,
    #         tgt_mask: Optional[Tensor] = None,
    #         memory_mask: Optional[Tensor] = None,
    #         tgt_key_padding_mask: Optional[Tensor] = None,
    #         memory_key_padding_mask: Optional[Tensor] = None,
    # ) -> Tensor:
    #     r"""Pass the inputs (and mask) through the decoder layer.
    #
    #     Args:
    #         tgt: the sequence to the decoder layer (required).
    #         memory: the sequence from the last layer of the encoder (required).
    #         tgt_mask: the mask for the tgt sequence (optional).
    #         memory_mask: the mask for the memory sequence (optional).
    #         tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
    #         memory_key_padding_mask: the mask for the memory keys per batch (optional).
    #
    #     Shape:
    #         see the docs in Transformer class.
    #     """
    #     tgt_norm = self.norm1(tgt)  # pre-norm
    #     tgt2 = self.self_attn(
    #         tgt_norm, tgt_norm, tgt_norm, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, freqs_cis=freqs_cis
    #     )[0]
    #     tgt = tgt + self.dropout1(tgt2)
    #
    #     tgt_norm = self.norm2(tgt)  # pre-norm
    #     tgt2, attn = self.multihead_attn(
    #         tgt_norm,
    #         memory,
    #         memory,
    #         arm=arm,
    #         attn_mask=memory_mask,
    #         key_padding_mask=memory_key_padding_mask,
    #     )
    #     tgt = tgt + self.dropout2(tgt2)
    #
    #     tgt_norm = self.norm3(tgt)  # pre-norm
    #     tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
    #     tgt = tgt + self.dropout3(tgt2)
    #     return tgt, attn

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            arm: Optional[AttentionRefinementModule],
            freqs_cis: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, freqs_cis=freqs_cis
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)  # post-norm
        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)  # post-norm
        tgt2, l_aux = self.ffn(tgt)
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)  # post-norm
        return tgt, attn, l_aux


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            num_layers: int,
            arm: Optional[AttentionRefinementModule],
            end: int,
            theta: float,
            norm=None,

    ):
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            copy.deepcopy(
                TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout
                ),
            )
            for _ in range(num_layers)
        ])
        self.norm = norm
        self.arm = arm

        # self.freqs_cis = precompute_freqs_cis(dim=d_model // nhead, end=end, theta=theta)
        # self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
            self,
            tgt: Tensor,
            memory: Tensor,
            height: int,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        # current_freqs_cis = self.freqs_cis[:tgt.size(0)]
        current_freqs_cis = None
        arm = None
        l_aux = 0.0
        for i, mod in enumerate(self.layers):
            output, attn, cur_l_aux = mod(
                output,
                memory,
                arm,
                freqs_cis=current_freqs_cis,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if i != len(self.layers) - 1 and self.arm is not None:
                arm = partial(self.arm, attn, memory_key_padding_mask, height)
            l_aux += cur_l_aux

        if self.norm is not None:
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
            dc: int,
            cross_coverage: bool,
            self_coverage: bool,
            end: int = 512,
            theta: float = 10000.0,
    ):
        super().__init__()

        self.word_embed = nn.Embedding(vocab_size, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.pos_enc = WordPosEnc(d_model=d_model)
        self.pos_norm = nn.LayerNorm(d_model)

        self.model = TransformerDecoder(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_layers=num_decoder_layers,
            arm=AttentionRefinementModule(
                nhead, dc, cross_coverage, self_coverage
            ) if (cross_coverage or self_coverage) else None,
            end=end,
            theta=theta,
        )

        self.proj = nn.Linear(d_model, vocab_size)

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
        """generate output for tgt

        Parameters
        ----------
        src : FloatTensor
            [b, h, w, d]
        src_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [b, l]

        Returns
        -------
        FloatTensor
            [b, l, vocab_size]
        """
        _, l = tgt.size()
        tgt_mask = self._build_attention_mask(l)
        tgt_pad_mask = tgt == vocab.PAD_IDX

        tgt = self.word_embed(tgt)
        tgt = self.norm(tgt)

        tgt = self.pos_enc(tgt)
        tgt = self.pos_norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out, l_aux = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

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
