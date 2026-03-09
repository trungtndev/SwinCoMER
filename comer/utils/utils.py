from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from comer.datamodule import vocab, vocab_size
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric

from pytorch_lightning.callbacks import Callback

SPECIAL_IDS = [
    vocab.SOS_IDX,
    vocab.EOS_IDX,
    vocab.PAD_IDX,
]
STRUCTURE_IDS = [
    vocab.word2idx[tok]
    for tok in [
        r"\frac",
        r"\sqrt",
        r"^",
        r"_",
        r"\limits",
        r"\sum",
        r"\int",
        "(", ")",
        "[", "]",
        "{", "}",
    ]
]
RADOM_PERTURB_TOKEN = [
    idx
    for idx in range(vocab_size) if idx not in SPECIAL_IDS
]
RADOM_PERTURB_TOKEN = torch.tensor(RADOM_PERTURB_TOKEN, dtype=torch.long)

NON_STRUCTURE_PERTURB_TOKEN = [
    idx
    for idx in range(vocab_size)
    if idx not in set(SPECIAL_IDS + STRUCTURE_IDS)
]
NON_STRUCTURE_PERTURB_TOKEN = torch.tensor(NON_STRUCTURE_PERTURB_TOKEN, dtype=torch.long)


class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
            self,
            seq_tensor: LongTensor,
            score: float,
            direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, indices_hat: List[List[int]], indices: List[List[int]]):
        for pred, truth in zip(indices_hat, indices):
            pred = vocab.indices2label(pred)
            truth = vocab.indices2label(truth)

            is_same = pred == truth

            if is_same:
                self.rec += 1

            self.total_line += 1

    def compute(self) -> float:
        exp_rate = self.rec / self.total_line
        return exp_rate


def ce_loss(
        output_hat: torch.Tensor,
        output: torch.Tensor,
        ignore_idx: int = vocab.PAD_IDX,
        reduction: str = "mean",
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx, reduction=reduction)
    return loss


def perturb_random(
        tokens: torch.Tensor,
        valid_ids_tensor: torch.Tensor = RADOM_PERTURB_TOKEN,
        prob: float = 0.0,
):
    if prob <= 0:
        return tokens
    new_token = tokens.clone()
    rand_mask = torch.rand(tokens.shape) < prob
    if not rand_mask.any():
        return new_token

    rand_idx = torch.randint(0, len(valid_ids_tensor), size=tokens.shape, )
    random_tokens = valid_ids_tensor[rand_idx]
    same_mask = random_tokens == tokens
    random_tokens[same_mask] = valid_ids_tensor[
        (rand_idx[same_mask] + 1) % len(valid_ids_tensor)
    ]
    new_token[rand_mask] = random_tokens[rand_mask]
    return new_token


def perturb_structure_random(
        tokens: torch.Tensor,
        prob: float = 0.0,
        structure_perturb_token=NON_STRUCTURE_PERTURB_TOKEN,
):
    if prob <= 0:
        return tokens

    new_tokens = tokens.clone()
    perturb_mask = torch.rand(tokens.shape) < prob

    if not perturb_mask.any():
        return new_tokens
    pool = structure_perturb_token
    rand_idx = torch.randint(0, len(pool), size=tokens.shape, )
    random_tokens = pool[rand_idx]
    # tránh random trùng chính nó
    same_mask = random_tokens == tokens
    random_tokens[same_mask] = pool[
        (rand_idx[same_mask] + 1) % len(pool)
        ]
    valid_mask = perturb_mask & torch.isin(tokens, pool)
    new_tokens[valid_mask] = random_tokens[valid_mask]
    return new_tokens

# perturb_mode: { none, radom, structure_random }
def to_tgt_output(
        tokens: Union[List[List[int]], List[LongTensor]],
        direction: str,
        device: torch.device,
        pad_to_len: Optional[int] = None,
        perturb_mode: Optional[str] = None,
        perturb_prob: float = 0.0,
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : Union[List[List[int]], List[LongTensor]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    if isinstance(tokens[0], list):
        tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]

    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]

    length = max(lens) + 1
    if pad_to_len is not None:
        length = max(length, pad_to_len)

    tgt = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, length),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        if perturb_mode == "random":
            tgt[i, 1: (1 + lens[i])] = perturb_random(token, perturb_prob)
        elif perturb_mode == "structure_random":
            tgt[i, 1: (1 + lens[i])] = perturb_structure_random(token, perturb_prob)
        else:
            tgt[i, 1: (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
        tokens: List[List[int]], device: torch.device, perturb_mode: str = "none", perturb_prob=0.0
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """
    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device, perturb_mode=perturb_mode, perturb_prob=perturb_prob)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device, perturb_mode=perturb_mode, perturb_prob=perturb_prob)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)

    return tgt, out
