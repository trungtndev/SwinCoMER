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

from comer.model.transformer.attention import MultiheadAttention, precompute_freqs_cis
from comer.model.transformer.arm import AttentionRefinementModule

from comer.utils.generation_utils import DecodeModel


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

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
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # print("freqs_cis", freqs_cis)
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask, freqs_cis=freqs_cis
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attn = self.multihead_attn(
            tgt,
            memory,
            memory,
            arm=arm,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            decoder_layer: TransformerDecoderLayer,
            num_layers: int,
            arm: Optional[AttentionRefinementModule],
            head_dim: int,
            end: int,
            theta: float,
            norm=None,
    ):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])

        self.num_layers = num_layers
        self.norm = norm

        self.arm = arm

        freqs_cis = precompute_freqs_cis(dim=head_dim, end=end, theta=theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

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

        arm = None
        for i, mod in enumerate(self.layers):
            output, attn = mod(
                output,
                memory,
                arm,
                freqs_cis=self.freqs_cis,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            if i != len(self.layers) - 1 and self.arm is not None:
                arm = partial(self.arm, attn, memory_key_padding_mask, height)

        if self.norm is not None:
            output = self.norm(output)

        return output

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
        self.word_norm = nn.LayerNorm(d_model)

        # self.pos_enc = WordPosEnc(d_model=d_model)
        # self.norm = nn.LayerNorm(d_model)

        arm = None
        if cross_coverage or self_coverage:
            arm = AttentionRefinementModule(nhead, dc, cross_coverage, self_coverage)

        self.model = TransformerDecoder(
            TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout),
            num_decoder_layers,
            arm,
            head_dim=d_model // nhead,
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

        tgt = self.word_embed(tgt)  # [b, l, d]
        tgt = self.word_norm(tgt)
        # tgt = self.pos_enc(tgt)  # [b, l, d]
        # tgt = self.norm(tgt)

        h = src.shape[1]
        src = rearrange(src, "b h w d -> (h w) b d")
        src_mask = rearrange(src_mask, "b h w -> b (h w)")
        tgt = rearrange(tgt, "b l d -> l b d")

        out = self.model(
            tgt=tgt,
            memory=src,
            height=h,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

        out = rearrange(out, "l b d -> b l d")
        out = self.proj(out)

        return out

    def transform(
            self, src: List[FloatTensor], src_mask: List[LongTensor], input_ids: LongTensor
    ) -> FloatTensor:
        assert len(src) == 1 and len(src_mask) == 1
        word_out = self(src[0], src_mask[0], input_ids)
        return word_out
