"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.
Giao diện:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers, seed)
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_DATASET_IMAGES = 17509


def load_split(labels_dir: str | Path, fold: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_path = Path(labels_dir)
    train_file = labels_path / f"train_subset{fold}.csv"
    val_file = labels_path / f"val_subset{fold}.csv"
    test_file = labels_path / f"test_subset{fold}.csv"

    if not train_file.exists() or not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy đủ file split fold {fold} trong {labels_dir}. "
            f"Cần có: {train_file.name}, {val_file.name}, {test_file.name}"
        )

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)

    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path | None = None) -> Dict[str, Any]:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    1. số ảnh mỗi tập và số ảnh mỗi lớp trong từng tập (kỳ vọng xấp xỉ 60/20/20)
    2. giao của từng cặp tập theo Filename phải RỖNG (train∩val, train∩test, val∩test)
    3. hợp ba tập phải bằng đúng 17.509 ảnh
    4. mọi Filename đều tồn tại trong `images_dir` (nếu images_dir tồn tại)
    """
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    total_n = n_train + n_val + n_test

    # 1. Kiểm tra tập hợp Filenames
    train_files = set(train_df["Filename"].tolist())
    val_files = set(val_df["Filename"].tolist())
    test_files = set(test_df["Filename"].tolist())

    inter_train_val = train_files.intersection(val_files)
    inter_train_test = train_files.intersection(test_files)
    inter_val_test = val_files.intersection(test_files)

    assert len(inter_train_val) == 0, f"Giao train và val không rỗng: {len(inter_train_val)} ảnh trùng"
    assert len(inter_train_test) == 0, f"Giao train và test không rỗng: {len(inter_train_test)} ảnh trùng"
    assert len(inter_val_test) == 0, f"Giao val và test không rỗng: {len(inter_val_test)} ảnh trùng"

    all_union = train_files.union(val_files).union(test_files)
    assert len(all_union) == TOTAL_DATASET_IMAGES, (
        f"Hợp 3 tập phải có đúng {TOTAL_DATASET_IMAGES} ảnh, thực tế có {len(all_union)}"
    )

    # 2. Phân phối theo lớp
    train_per_class = train_df["Label"].value_counts().sort_index().to_dict()
    val_per_class = val_df["Label"].value_counts().sort_index().to_dict()
    test_per_class = test_df["Label"].value_counts().sort_index().to_dict()

    # 3. Kiểm tra file tồn tại trên đĩa nếu images_dir được cung cấp và tồn tại
    missing_files = 0
    if images_dir is not None and Path(images_dir).exists():
        img_path = Path(images_dir)
        for fname in all_union:
            if not (img_path / fname).exists():
                missing_files += 1
        if missing_files > 0:
            raise FileNotFoundError(f"Có {missing_files} file ảnh trong split không tồn tại ở {images_dir}")

    summary = {
        "n": {
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "total": total_n,
            "ratio": f"{n_train/total_n*100:.1f}/{n_val/total_n*100:.1f}/{n_test/total_n*100:.1f}"
        },
        "per_class": {
            "train": train_per_class,
            "val": val_per_class,
            "test": test_per_class
        },
        "overlap": {
            "train_val": len(inter_train_val),
            "train_test": len(inter_train_test),
            "val_test": len(inter_val_test)
        },
        "missing_files": missing_files
    }
    return summary


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic") -> T.Compose:
    """Tạo torchvision transforms.

    Train:
      - "basic": RandomResizedCrop(img_size) + RandomHorizontalFlip()
      - "color": basic + ColorJitter
      - "randaug": basic + RandAugment
      - "trivial": basic + TrivialAugmentWide
      - "flip_vh": basic + RandomVerticalFlip
    Val/Test:
      - Resize(256) -> CenterCrop(img_size) nếu img_size < 256; nếu img_size == 256 thì Resize((256, 256))
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if not train:
        if img_size == 256:
            return T.Compose([
                T.Resize((256, 256)),
                T.ToTensor(),
                normalize,
            ])
        else:
            return T.Compose([
                T.Resize(256),
                T.CenterCrop(img_size),
                T.ToTensor(),
                normalize,
            ])

    # Chế độ huấn luyện (train=True)
    t_list = []
    t_list.append(T.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
    t_list.append(T.RandomHorizontalFlip(p=0.5))

    if aug == "basic":
        pass
    elif aug == "color":
        t_list.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    elif aug == "randaug":
        t_list.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "trivial":
        t_list.append(T.TrivialAugmentWide())
    elif aug == "flip_vh":
        t_list.append(T.RandomVerticalFlip(p=0.5))
    else:
        pass

    t_list.extend([
        T.ToTensor(),
        normalize
    ])
    return T.Compose(t_list)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform: Tensor, nhãn: int, tên file: str).
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, int, str]:
        fname = self.filenames[i]
        label = self.labels[i]
        img_path = self.images_dir / fname

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Lỗi khi đọc file ảnh {img_path}: {e}")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, fname


def _seed_worker(worker_id: int) -> None:
    """Đảm bảo worker của DataLoader có seed độc lập và tái lập được (Slide 59)."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0) -> DataLoader:
    """Tạo DataLoader cho tập dữ liệu DeepWeeds với worker seed cố định."""
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)

    sampler_obj = None
    shuffle = False

    if train:
        if sampler == "balanced":
            class_counts = df["Label"].value_counts().to_dict()
            class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
            sample_weights = [class_weights_dict[lbl] for lbl in df["Label"]]
            sampler_obj = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        else:
            shuffle = True
    else:
        shuffle = False

    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    return loader
