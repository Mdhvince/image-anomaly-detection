from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import SubsetRandomSampler


def generate_dataframe_from_images(data_root: Path) -> pd.DataFrame:
    rows = []
    for category_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        for split in ("train", "test"):
            for defect_dir in sorted((category_dir / split).iterdir()):
                label = 0.0 if defect_dir.name == "good" else 1.0
                for image_path in sorted(defect_dir.glob("*.png")):
                    mask_path = ""
                    if label == 1.0:
                        mask_path = str(category_dir / "ground_truth" / defect_dir.name / f"{image_path.stem}_mask.png")
                    rows.append({
                        "category": category_dir.name,
                        "split": split,
                        "label": label,
                        "image_path": str(image_path),
                        "mask_path": mask_path,
                    })
    data = pd.DataFrame.from_records(rows)
    return data


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
