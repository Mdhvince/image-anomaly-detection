"""Real-condition inference: score ONE image file with the trained multi-class model.

Prints the image-level anomaly score, the binary decision when the checkpoint
carries a threshold, and displays a side-by-side figure (image | anomaly map overlay).

Run: python inference_one_image.py path/to/image.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from config import checkpoint_path, get_device, load_config
from inference import compute_anomaly_maps, image_scores
from model import build_model
from preprocess import apply_preprocessing, denormalize


def predict_anomaly(model: torch.nn.Module, config: dict, image_path: Path) -> tuple:
    """One image file -> (anomaly map, image-level score, preprocessed image for display).

    :param model: trained Dinomaly model in eval mode.
    :param config: configuration dict (load_config).
    :param image_path: image file to score.
    :return: (anomaly_map (H, W), score, prepared image tensor (3, H, W)).
    """
    prepared = apply_preprocessing(
        Image.open(image_path).convert("RGB"), config["img_size"], config["crop_size"]
    ).unsqueeze(0)
    anomaly_maps = compute_anomaly_maps(model, prepared.to(next(model.parameters()).device), prepared.shape[-1])
    score = float(image_scores(anomaly_maps, config["top_pixel_ratio"])[0])
    return anomaly_maps[0, 0].cpu(), score, prepared[0]


def main() -> None:
    """Load the checkpoint, score the image given as CLI argument, show the decision figure."""
    if len(sys.argv) != 2:
        sys.exit(f"usage: python {Path(__file__).name} <image_path>")
    image_path = Path(sys.argv[1])

    config = load_config()
    model = build_model(config, get_device())
    checkpoint = torch.load(checkpoint_path())
    model.bottleneck.load_state_dict(checkpoint["bottleneck"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()                                                   # dropout off: deterministic maps
    threshold = checkpoint.get("threshold")

    anomaly_map, score, prepared = predict_anomaly(model, config, image_path)
    decision = "" if threshold is None else f" -> {'ANOMALIE' if score >= threshold else 'normale'} (seuil {threshold:.4f})"

    original = denormalize(prepared).permute(1, 2, 0)
    map_display = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    figure, (axis_image, axis_map) = plt.subplots(1, 2, figsize=(9, 4.5))
    axis_image.imshow(original)
    axis_image.set_title("image")
    axis_image.axis("off")
    axis_map.imshow(original)
    axis_map.imshow(map_display, cmap="jet", alpha=0.5)
    axis_map.set_title("carte d'anomalie")
    axis_map.axis("off")
    figure.suptitle(f"score: {score:.4f}{decision or ' (plus haut = plus anormal)'}", fontsize=11)
    figure.tight_layout()

    print(f"{image_path.name}: score {score:.4f}{decision}")
    plt.show()


if __name__ == "__main__":
    main()
