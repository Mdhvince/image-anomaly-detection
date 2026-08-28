"""Project configuration: raw dictionaries, no parsing - edit values directly here.

CONFIG holds the paper regimen (MVTec-AD, multi-class). DEMO holds the lighter
regimen (same mechanics) and is merged over CONFIG when USE_DEMO is enabled.
"""

from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "datasets" / "mvtec_anomaly_detection"
INDEX_CSV = DATA_ROOT / "index.csv"

CONFIG = {
    # data
    "data_root": DATA_ROOT,
    "index_csv": INDEX_CSV,
    # model
    "backbone": "dinov2_vitb14_reg",
    "target_layers": [2, 3, 4, 5, 6, 7, 8, 9],  # 8 middle layers out of 12
    "embed_dim": 768,
    "drop_rate": 0.2,
    "groups": [[0, 1, 2, 3], [4, 5, 6, 7]],  # low level; high level
    # preprocessing: resize 448, center crop 392 (392 / 14 = 28x28 tokens)
    "img_size": 448,
    "crop_size": 392,
    # loaders
    "batch_size": 16,
    "num_workers": 4,
    # training; num_iterations is the paper budget, converted to epochs (ceil over batches per epoch)
    "num_iterations": 10000,
    "valid_ratio": 0.2,  # fraction of the normal train images held out for validation
    "learning_rate": 2e-3,
    "final_learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "warmup_iterations": 100,
    "max_grad_norm": 0.1,
    # hard mining: fraction of easy points whose gradient is reduced (the sg(.)_0.1 of the paper)
    "final_mining_percent": 0.9,
    "mining_ramp_iterations": 1000,
    "shrink_factor": 0.1,
    # evaluation: image score = mean of the top 1% of the map
    "top_pixel_ratio": 0.01,
}

# Lighter regimen for quick checks; set USE_DEMO = False for the paper regimen.
DEMO = {
    "img_size": 224,
    "crop_size": 224,
    "batch_size": 4,
    "num_iterations": 1500,
    "mining_ramp_iterations": 150,
}
USE_DEMO = True


def get_config() -> dict:
    """Full configuration dict, DEMO regimen merged over CONFIG when enabled.

    :return: configuration dict shared by every script.
    """
    return CONFIG | DEMO if USE_DEMO else CONFIG


def get_device() -> torch.device:
    """Best available accelerator.

    :return: cuda if available, then mps, else cpu.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path() -> Path:
    """Checkpoint file of the single multi-class model (bottleneck + decoder weights), parent created.

    :return: path of the checkpoint file.
    """
    path = PROJECT_ROOT / "checkpoints" / "dinomaly.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    for key, value in get_config().items():
        print(f"{key} = {value}")
