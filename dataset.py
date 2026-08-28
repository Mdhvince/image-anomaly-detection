from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import SubsetRandomSampler


def index_mvtec(data_root: Path, csv_path: Path) -> int:
    """Walk the MVTec-AD tree (every category, both splits) and write the flat index CSV.

    Expect the official layout - this is the ONLY place that knows it:
    category/train/good, category/test/good + test/<defect>,
    category/ground_truth/<defect>/<image name>_mask.png.

    :param data_root: MVTec-AD root folder containing one subfolder per category.
    :param csv_path: where the flat index is written (overwritten every run).
    :return: number of indexed images.
    """
    rows = []
    for category_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for split in ("train", "test"):
            for defect_dir in sorted((category_dir / split).iterdir()):
                label = 0.0 if defect_dir.name == "good" else 1.0
                for image_path in sorted(defect_dir.glob("*.png")):
                    mask_path = ""
                    if label == 1.0:
                        mask_path = str(
                            (category_dir / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png").relative_to(category_dir)
                        )
                    rows.append({
                        "category": category_dir.name,
                        "split": split,
                        "label": label,
                        "image_path": str(image_path.relative_to(category_dir)),
                        "mask_path": mask_path,
                    })
    pd.DataFrame.from_records(rows).to_csv(csv_path, index=False)
    return len(rows)


class MVTecDataset(torch.utils.data.Dataset):
    """Dataset from the flat index CSV - Expect one row per image with columns
    category,split,label,image_path,mask_path (schema in docs/method.md).
    """

    def __init__(self, index_rows: list, data_root: Path, img_size: int, crop_size: int, preprocess: Callable) -> None:
        self.preprocess = preprocess
        self.img_size, self.crop_size = img_size, crop_size
        self.samples = [
            (
                data_root / row["category"] / row["image_path"],
                data_root / row["category"] / row["mask_path"] if row["mask_path"] else None,
                float(row["label"]),
            )
            for row in index_rows
        ]
        self.labels = [label for _, _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        image_path, mask_path, label = self.samples[index]
        image = self.preprocess(Image.open(image_path).convert("RGB"), self.img_size, self.crop_size)
        if mask_path is None:
            mask = torch.zeros(1, self.crop_size, self.crop_size)
        else:
            mask = self.preprocess(Image.open(mask_path).convert("L"), self.img_size, self.crop_size, is_mask=True)
        return image, mask, label


def train_valid_split(training_set: torch.utils.data.Dataset, valid_ratio: float) -> tuple:
    """SubsetRandomSamplers over the train/validation indices of one dataset.

    :param training_set: dataset to split.
    :param valid_ratio: fraction of the samples reserved for validation.
    :return: (train_sampler, valid_sampler) over disjoint shuffled indices.
    """
    num_train = len(training_set)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_ratio * num_train))
    train_indices, valid_indices = indices[split:], indices[:split]
    return SubsetRandomSampler(train_indices), SubsetRandomSampler(valid_indices)


def build_loaders(train_set: torch.utils.data.Dataset, test_set: torch.utils.data.Dataset, batch_size: int, valid_ratio: float, num_workers: int) -> tuple:
    """Build the training, validation and evaluation DataLoaders.

    :param train_set: normal-only training images, further split into train/validation.
    :param test_set: mixed test images (normals + anomalies with masks).
    :param batch_size: batch size of both loaders.
    :param valid_ratio: fraction of train_set reserved for validation.
    :param num_workers: subprocesses of the train/validation loaders.
    :return: (train_loader, valid_loader, test_loader); training drops its last batch, validation and evaluation are ordered by sampler.
    """
    train_sampler, valid_sampler = train_valid_split(train_set, valid_ratio)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, drop_last=True)
    valid_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def list_categories(index_csv: Path) -> list:
    """Sorted MVTec-AD categories of the flat index.

    :param index_csv: flat index produced by index_mvtec.
    :return: sorted unique category names (ex: ["bottle", "cable", ...]).
    """
    return sorted(pd.read_csv(index_csv, usecols=["category"]).category.unique())


def build_dataloaders(index_csv: Path, img_size: int, crop_size: int, batch_size: int, valid_ratio: float, num_workers: int, preprocess: Callable, category_name: str = None) -> tuple:
    """Index CSV -> loaders: every category merged by default (one model for all), restricted to one category if set.

    :param index_csv: flat index produced by index_mvtec.
    :param img_size: preprocessing resize size.
    :param crop_size: preprocessing crop size.
    :param batch_size: batch size of both loaders.
    :param valid_ratio: fraction of the normal train images reserved for validation.
    :param num_workers: subprocesses of the train/validation loaders.
    :param preprocess: image pipeline (ex: apply_preprocessing).
    :param category_name: none (train on all categories) or one MVTec-AD category (per-category evaluation).
    :return: (train_loader, valid_loader, test_loader, test_set).
    """
    index_frame = pd.read_csv(index_csv, keep_default_na=False, dtype={"label": float})
    category_rows = index_frame if category_name is None else index_frame[index_frame.category == category_name]

    train_rows = category_rows[category_rows.split == "train"].to_dict("records")
    test_rows = category_rows[category_rows.split == "test"].to_dict("records")

    train_set = MVTecDataset(train_rows, index_csv.parent, img_size, crop_size, preprocess)
    test_set = MVTecDataset(test_rows, index_csv.parent, img_size, crop_size, preprocess)
    train_loader, valid_loader, test_loader = build_loaders(train_set, test_set, batch_size, valid_ratio, num_workers)
    return train_loader, valid_loader, test_loader, test_set
