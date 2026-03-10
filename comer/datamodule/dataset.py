import torchvision.transforms as tr
from torch.utils.data.dataset import Dataset
import numpy as np
import random
from PIL import Image
import albumentations as A
import os
from .transforms import AlbScaleAugmentation, ScaleToLimitRange, ScaleAugmentation, ResizeLimit

K_MIN = 0.7
K_MAX = 1.4

H_LO = 16
H_HI = 256
W_LO = 16
W_HI = 1024


def find_image_path(base):
    for ext in [".bmp", ".png", ".jpg", ".jpeg"]:
        path = base + ext
        if os.path.exists(path):
            return path
    raise FileNotFoundError(base)


# class CROHMEDataset(Dataset):
#     def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
#         super().__init__()
#         self.ds = [
#             (fname, find_image_path(p), caption)
#             for fname, p, caption in ds
#         ]
#
#         trans_list = []
#         if is_train and scale_aug:
#             trans_list.append(AlbScaleAugmentation(K_MIN, K_MAX))
#
#         trans_list += [
#             ResizeLimit(height=256, width=512),
#             A.PadIfNeeded(
#                 min_height=256,
#                 min_width=512,
#                 fill=0,
#                 position="center"
#             ),
#             A.ToRGB(),
#             A.Normalize(
#                 mean=(0.485, 0.456, 0.406),
#                 std=(0.229, 0.224, 0.225)
#             ),
#             A.ToTensorV2(),
#
#         ]
#         self.transform = A.Compose(trans_list)
#
#     def __getitem__(self, idx):
#         fname, p, caption = self.ds[idx]
#
#         img = Image.open(p)
#         img = self.transform(image=np.array(img))["image"]
#
#         return fname, img, caption
#
#     def __len__(self):
#         return len(self.ds)

class CROHMEDataset(Dataset):
    def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
        super().__init__()
        self.ds = [
            (fname, find_image_path(p), caption)
            for fname, p, caption in ds
        ]
        trans_list = []
        if is_train and scale_aug:
            trans_list.append(ScaleAugmentation(K_MIN, K_MAX))

        trans_list += [
            ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
            tr.ToTensor(),
        ]
        self.transform = tr.Compose(trans_list)

    def __getitem__(self, idx):
        fname, p, caption = self.ds[idx]

        # img = [self.transform(im) for im in img]
        img = Image.open(p)
        img = self.transform(np.array(img))

        return fname, img, caption

    def __len__(self):
        return len(self.ds)

# class CROHMEDataset(Dataset):
#     def __init__(self, ds, is_train: bool, scale_aug: bool) -> None:
#         super().__init__()
#         self.ds = ds
#         self.is_train = is_train
#
#         trans_list = []
#         if is_train and scale_aug:
#             trans_list.append(ScaleAugmentation(K_MIN, K_MAX))
#
#         trans_list += [
#             ScaleToLimitRange(w_lo=W_LO, w_hi=W_HI, h_lo=H_LO, h_hi=H_HI),
#             tr.ToTensor(),
#         ]
#         self.transform = tr.Compose(trans_list)
#
#         if self.is_train:
#             self.caption_to_indices = {}
#             for idx, item in enumerate(self.ds):
#                 caption = item[2]
#
#                 if isinstance(caption, list):
#                     caption = tuple(caption)
#
#                 if caption not in self.caption_to_indices:
#                     self.caption_to_indices[caption] = []
#                 self.caption_to_indices[caption].append(idx)
#
#             self.unique_captions = list(self.caption_to_indices.keys())
#
#     def __getitem__(self, idx):
#         if self.is_train:
#             target_caption = self.unique_captions[idx]
#             chosen_idx = random.choice(self.caption_to_indices[target_caption])
#             fname, img, caption = self.ds[chosen_idx]
#         else:
#             fname, img, caption = self.ds[idx]
#
#         img = self.transform(np.array(img))
#
#         return fname, img, caption
#
#     def __len__(self):
#         if self.is_train:
#             return len(self.unique_captions)
#
#         return len(self.ds)
