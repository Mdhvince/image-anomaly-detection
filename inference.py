from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from config import checkpoint_path, get_device, load_config
from dataset import build_dataloaders, index_mvtec, list_categories
from model import build_model
from preprocess import apply_preprocessing


@torch.no_grad()
def compute_anomaly_maps(model: torch.nn.Module, images: torch.Tensor, map_size: int) -> torch.Tensor:
    """Per-point cosine distance between (encoder, decoder) groups, upsampled, averaged.

    :param model: trained Dinomaly model.
    :param images: batch of preprocessed images.
    :param map_size: edge of the square output maps.
    :return: anomaly maps, one per image, (B, 1, map_size, map_size).
    """
    encoder_groups, decoder_groups = model(images)
    group_maps = []
    for encoder_group, decoder_group in zip(encoder_groups, decoder_groups):
        distance_map = 1 - F.cosine_similarity(encoder_group, decoder_group, dim=-1)  # (B, N)
        grid_side = int(distance_map.shape[1] ** 0.5)
        distance_map = distance_map.reshape(-1, 1, grid_side, grid_side)
        distance_map = F.interpolate(distance_map, size=map_size, mode="bilinear", align_corners=True)
        group_maps.append(distance_map)
    return torch.cat(group_maps, dim=1).mean(dim=1, keepdim=True)  # (B, 1, H, W)


def image_scores(anomaly_maps: torch.Tensor, top_pixel_ratio: float = 0.01) -> torch.Tensor:
    """Image score = mean of the top top_pixel_ratio% values of the map (robust max).

    :param anomaly_maps: (B, 1, H, W) anomaly maps.
    :param top_pixel_ratio: fraction of the highest pixels averaged into the score.
    :return: one score per image.
    """
    flat_maps = anomaly_maps.flatten(1)
    num_top_values = max(1, int(flat_maps.shape[1] * top_pixel_ratio))
    return flat_maps.sort(dim=1, descending=True)[0][:, :num_top_values].mean(dim=1)


def evaluate_image_auroc(model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, top_pixel_ratio: float) -> float:
    """Image-level AUROC over the whole test loader (model set to eval mode).

    :param model: trained Dinomaly model.
    :param test_loader: ordered evaluation DataLoader.
    :param top_pixel_ratio: passed to image_scores.
    :return: I-AUROC between 0 and 1.
    """
    model.eval()
    device = next(model.parameters()).device
    score_batches, label_batches = [], []
    for images, _, labels in test_loader:
        anomaly_maps = compute_anomaly_maps(model, images.to(device), images.shape[-1])
        score_batches.append(image_scores(anomaly_maps, top_pixel_ratio).cpu())
        label_batches.append(labels)
    return roc_auc_score(torch.cat(label_batches).numpy(), torch.cat(score_batches).numpy())


def main() -> None:
    """For every category: rebuild the model, load its checkpoint, print the image-level AUROC (mean at the end)."""
    config = load_config()
    device = get_device()

    data_root = Path(config["data_root"])
    index_csv = data_root / "index.csv"
    index_mvtec(data_root, index_csv)

    model = build_model(config, device)
    checkpoint = torch.load(checkpoint_path())
    model.bottleneck.load_state_dict(checkpoint["bottleneck"])
    model.decoder.load_state_dict(checkpoint["decoder"])

    aurocs = []
    categories = list_categories(index_csv)
    for category_name in categories:
        _, _, test_loader, test_set = build_dataloaders(
            index_csv, config["img_size"], config["crop_size"],
            config["batch_size"], config["valid_ratio"], config["num_workers"], apply_preprocessing,
            category_name=category_name,
        )
        auroc = evaluate_image_auroc(model, test_loader, config["top_pixel_ratio"])
        aurocs.append(auroc)
        print(f"{category_name}: image-level AUROC {auroc:.3f} ({int(sum(test_set.labels))} anomalies / {len(test_set)} images)")
    print(f"Mean image-level AUROC over {len(categories)} categories: {sum(aurocs) / len(aurocs):.3f}")


if __name__ == "__main__":
    main()
