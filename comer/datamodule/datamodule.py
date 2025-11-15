import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm

from comer.datamodule.dataset import CROHMEDataset
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data.dataloader import DataLoader

# Data = List[Tuple[str, Image.Image, List[str]]]
Data = List[Tuple[str, Tuple[int, int], List[str]]]

MAX_SIZE = 32e4  # change here accroading to your GPU memory


def extract_data(archive: str, dir_name: str) -> Data:
    """Extract all data need for a dataset from zip archive

    Args:
        archive (ZipFile):
        dir_name (str): dir name in archive zip (eg: train, test_2014......)

    Returns:
        Data: list of tuple of image and formula
    """
    # with archive.open(f"data/{dir_name}/caption.txt", "r") as f:
    with open(f"{archive}/{dir_name}/caption.txt", "rb") as f:
        captions = f.readlines()
    data = []
    for line in tqdm(captions):
        tmp = line.decode().strip().split()
        img_name = tmp[0]
        formula = tmp[1:]

        img_path = f"{archive}/{dir_name}/img/{img_name}.bmp"
        data.append((img_path, formula))

    print(f"Extract data from: {dir_name}, with data size: {len(data)}")

    return data


@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    seq: List[str]  # [b,]

    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            seq=self.seq,
        )


# @dataclass
# class Batch:
#     img_bases: List[str]  # [b,]
#     imgs: FloatTensor  # [b, 1, H, W]
#     mask: LongTensor  # [b, H, W]
#     indices: List[List[int]]  # [b, l]
#
#     def __len__(self) -> int:
#         return len(self.img_bases)
#
#     def to(self, device) -> "Batch":
#         return Batch(
#             img_bases=self.img_bases,
#             imgs=self.imgs.to(device),
#             mask=self.mask.to(device),
#             indices=self.indices,
#         )

# def collate_fn(batch):
#     # assert len(batch) == 1
#     # batch = batch[0]
#     fnames = [item[0] for item in batch]
#     images_x = [item[1] for item in batch]
#     seqs_y = [vocab.words2indices(item[2]) for item in batch]
#
#     heights_x = [s.size(1) for s in images_x]
#     widths_x = [s.size(2) for s in images_x]
#
#     n_samples = len(heights_x)
#     max_height_x = max(heights_x)
#     max_width_x = max(widths_x)
#
#     x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
#     x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
#     for idx, s_x in enumerate(images_x):
#         x[idx, :, : heights_x[idx], : widths_x[idx]] = s_x
#         x_mask[idx, : heights_x[idx], : widths_x[idx]] = 0
#
#     # return fnames, x, x_mask, seqs_y
#     return Batch(fnames, x, x_mask, seqs_y)

def collate_fn(batch):
    # batch = [(fname, img_tensor, caption), ...]
    img_bases = [item[0] for item in batch]
    imgs = torch.stack([item[1] for item in batch], dim=0)
    seq = [item[2] for item in batch]
    B, C, H, W = imgs.size()

    mask = torch.zeros(B, H, W, dtype=torch.bool)

    return Batch(
        img_bases=img_bases,
        imgs=imgs,
        mask=mask,
        seq=seq
    )


class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        zipfile_path: str = f"{os.path.dirname(os.path.realpath(__file__))}/../../data.zip",
        test_year: str = "2014",
        train_batch_size: int = 8,
        eval_batch_size: int = 4,
        num_workers: int = 5,
        scale_aug: bool = False,
    ) -> None:
        super().__init__()
        assert isinstance(test_year, str)
        self.zipfile_path = zipfile_path
        self.test_year = test_year
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.scale_aug = scale_aug

        print(f"Load data from: {self.zipfile_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        # with ZipFile(self.zipfile_path) as archive:
            if stage == "fit" or stage is None:
                self.train_dataset = CROHMEDataset(
                    # build_dataset(self.zipfile_path, "train", self.train_batch_size),
                    extract_data(self.zipfile_path, "train"),
                    True,
                    self.scale_aug,
                )
                self.val_dataset = CROHMEDataset(
                    # build_dataset(self.zipfile_path, self.test_year, self.eval_batch_size),
                    extract_data(self.zipfile_path, self.test_year),
                    False,
                    self.scale_aug,
                )
            if stage == "test" or stage is None:
                self.test_dataset = CROHMEDataset(
                    # build_dataset(self.zipfile_path, self.test_year, self.eval_batch_size),
                    extract_data(self.zipfile_path, self.test_year),
                    False,
                    self.scale_aug,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )
