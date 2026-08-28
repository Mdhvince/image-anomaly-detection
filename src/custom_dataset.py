"""Dataset over slices of the flat data CSV - the CSV already carries full paths."""
from collections.abc import Callable

import pandas as pd
import torch
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    """One row of data.csv per sample; this class only loads fields, never assembles paths.

    Expected DataFrame columns (produced by dataset_utils.build_data):
    category, split, label, image_path (absolute), mask_path (absolute, "" when none).

    :param data: DataFrame slice of the data CSV (ex: all train rows, or one category's test rows).
    :param img_size: preprocessing resize size.
    :param crop_size: preprocessing crop size.
    :param preprocess: image pipeline (ex: preprocess.apply_preprocessing).
    """

    def __init__(self, data: pd.DataFrame, img_size: int, crop_size: int, preprocess: Callable) -> None:
        self.data = data.reset_index(drop=True)
        self.img_size = img_size
        self.crop_size = crop_size
        self.preprocess = preprocess
        self.labels = self.data["label"].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        row = self.data.iloc[index]
        image = self.preprocess(Image.open(row.image_path).convert("RGB"), self.img_size, self.crop_size)
        if row.mask_path == "":
            mask = torch.zeros(1, self.crop_size, self.crop_size)
        else:
            mask = self.preprocess(Image.open(row.mask_path).convert("L"), self.img_size, self.crop_size, is_mask=True)
        return image, mask, float(row.label)
