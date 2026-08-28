"""Display helpers: dataset preview, preprocessing check, loss curve, anomaly maps."""
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from inference import compute_anomaly_maps


def plot_test_preview(test_set: torch.utils.data.Dataset, denormalize: Callable, title: str) -> plt.Figure:
    """First 2 anomalies (defect tinted) then 2 normal test images.

    :param test_set: CustomDataset with .labels and .data attributes.
    :param denormalize: inverse of the ImageNet normalization (preprocess.denormalize).
    :param title: figure title (ex: the category name).
    :return: the figure.
    """
    anomalous_indices = [index for index, label in enumerate(test_set.labels) if label == 1.0][:2]
    normal_indices = [index for index, label in enumerate(test_set.labels) if label == 0.0][:2]
    preview_samples = [test_set[sample_index] for sample_index in anomalous_indices + normal_indices]
    images = torch.stack([sample[0] for sample in preview_samples])
    masks = torch.stack([sample[1] for sample in preview_samples])
    labels = [sample[2] for sample in preview_samples]

    figure, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for axis, image, mask, label in zip(axes, images, masks, labels):
        display_image = denormalize(image).permute(1, 2, 0).numpy().copy()
        is_defective = mask[0].numpy() > 0.5
        display_image[is_defective] = display_image[is_defective] * 0.35 + np.array([1.0, 0.25, 0.25]) * 0.65
        axis.imshow(display_image)
        axis.set_title(f"anomalie: {bool(label)}", fontsize=9)
        axis.axis("off")
    figure.suptitle(title, fontsize=10)
    return figure


def plot_preprocessing_check(test_set: torch.utils.data.Dataset, img_size: int, crop_size: int, apply_preprocessing: Callable, denormalize: Callable) -> plt.Figure:
    """One normal and one anomalous test image: raw input vs preprocessed tensor vs mask.

    :param test_set: CustomDataset with .labels and .data attributes.
    :param img_size: preprocessing resize size.
    :param crop_size: preprocessing crop size.
    :param apply_preprocessing: image pipeline (preprocess.apply_preprocessing).
    :param denormalize: inverse of the ImageNet normalization (preprocess.denormalize).
    :return: the figure.
    """
    anomalous_index = test_set.labels.index(1.0)
    normal_index = test_set.labels.index(0.0)

    figure, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    for row_index, sample_index in enumerate((normal_index, anomalous_index)):
        row = test_set.data.iloc[sample_index]
        raw_image = Image.open(row.image_path).convert("RGB")
        image = apply_preprocessing(raw_image, img_size, crop_size)
        if row.mask_path == "":
            mask = torch.zeros(1, crop_size, crop_size)
        else:
            mask = apply_preprocessing(Image.open(row.mask_path).convert("L"), img_size, crop_size, is_mask=True)
        label = float(row.label)

        axes[row_index, 0].imshow(raw_image)
        axes[row_index, 0].set_title(f"image brute (label {label:.0f})", fontsize=9)
        axes[row_index, 1].imshow(denormalize(image).permute(1, 2, 0).numpy())
        axes[row_index, 1].set_title(
            f"apres preprocessing {tuple(image.shape)}, valeurs [{image.min():.2f}, {image.max():.2f}]", fontsize=9
        )
        axes[row_index, 2].imshow(mask[0], cmap="gray", vmin=0, vmax=1)
        axes[row_index, 2].set_title(f"masque {tuple(mask.shape)}", fontsize=9)
        for axis in axes[row_index]:
            axis.axis("off")
    figure.suptitle(
        "Ce que le modele mange vraiment: resize -> crop central -> normalisation ImageNet "
        "(valeurs hors [0, 1], denormalisees ici pour l'affichage)",
        fontsize=10,
    )
    return figure


def plot_anomaly_maps(model: torch.nn.Module, loader: torch.utils.data.DataLoader, denormalize: Callable, max_rows: int = 6, draw_contour: bool = False) -> plt.Figure:
    """First batch: image / ground-truth mask / anomaly map side by side.

    With draw_contour=True the image column also carries the anomaly contour:
    iso-line of the map at its 99th percentile, i.e. the boundary of the top 1%
    pixels (same fraction as the image score, config top_pixel_ratio).

    :param model: trained Dinomaly model.
    :param loader: evaluation DataLoader.
    :param denormalize: inverse of the ImageNet normalization (preprocess.denormalize).
    :param max_rows: maximum number of rows displayed.
    :param draw_contour: overlay the top-1% anomaly contour on the image column.
    :return: the figure.
    """
    device = next(model.parameters()).device
    images, masks, labels = next(iter(loader))
    anomaly_maps = compute_anomaly_maps(model, images.to(device), images.shape[-1])
    num_rows_shown = min(max_rows, images.shape[0])

    figure, axes = plt.subplots(num_rows_shown, 3, figsize=(8, 2.3 * num_rows_shown))
    axes = axes.reshape(num_rows_shown, 3)
    for row_index in range(num_rows_shown):
        anomaly_map = anomaly_maps[row_index, 0].cpu().numpy()
        axes[row_index, 0].imshow(denormalize(images[row_index]).permute(1, 2, 0).numpy())
        axes[row_index, 0].set_title(f"image (anomalie: {bool(labels[row_index])})", fontsize=8)
        if draw_contour:
            axes[row_index, 0].contour(anomaly_map, levels=[np.quantile(anomaly_map, 0.99)], colors="red", linewidths=1)
        axes[row_index, 1].imshow(masks[row_index, 0], cmap="gray")
        axes[row_index, 1].set_title("masque GT", fontsize=8)
        axes[row_index, 2].imshow(anomaly_map, cmap="inferno")
        axes[row_index, 2].set_title("carte d'anomalie", fontsize=8)
        for axis in axes[row_index]:
            axis.axis("off")
    figure.tight_layout()
    return figure
