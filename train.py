import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from config import checkpoint_path, get_device, load_config
from dataset import build_dataloaders, index_mvtec
from inference import compute_normal_threshold
from loss import build_optimizer, global_cosine_hm
from model import build_model
from preprocess import apply_preprocessing, denormalize


def train_dinomaly(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR, config: dict, save_path: Path) -> None:
    """Epoch loop: train on normal images, validate on held-out normals, save on validation improvement.

    The mining rate ramps up linearly from 0 to final_mining_percent over
    mining_ramp_iters iterations; warmup/cosine schedule and the 0.1 gradient
    clip follow the paper's iteration count. Losses and the first training
    batch are logged to TensorBoard (runs/).

    :param model: assembled Dinomaly model.
    :param train_loader: DataLoader over the normal train split.
    :param valid_loader: DataLoader over the held-out normal validation split.
    :param optimizer: AdamW on the trainable modules.
    :param scheduler: warmup/cosine schedule, stepped each iteration.
    :param config: configuration dict (load_config).
    :param save_path: checkpoint file rewritten each time the validation loss decreases.
    """
    writer = SummaryWriter()
    device = next(model.parameters()).device
    num_epochs = math.ceil(config["num_iterations"] / len(train_loader))
    valid_loss_min = float("inf")
    iteration = 0

    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for images, _, _ in train_loader:
            images = images.to(device)
            iteration += 1
            if iteration == 1:
                writer.add_image("training/first_batch", make_grid(denormalize(images.cpu()), nrow=4), iteration)
            mining_percent = min(                                   # hard mining rate: 0 -> final_mining_percent
                config["final_mining_percent"] * iteration / config["mining_ramp_iters"],
                config["final_mining_percent"],
            )
            optimizer.zero_grad()                                   # Clear the gradients, they accumulate at each step
            encoder_groups, decoder_groups = model(images)          # Forward through the frozen encoder
            loss = global_cosine_hm(                                # Compute the loss (+ register the mining hooks)
                encoder_groups, decoder_groups, mining_percent, config["shrink_factor"]
            )
            loss.backward()                                         # Compute the gradients (easy points: grad x0.1)
            nn.utils.clip_grad_norm_(                               # Clip the gradients (paper: max_norm 0.1)
                [
                    parameter
                    for module in (model.bottleneck, model.decoder)
                    for parameter in module.parameters()
                ],
                max_norm=config["max_grad_norm"],
            )
            optimizer.step()                                        # Perform updates using calculated gradients
            scheduler.step()
            train_loss += loss.item() * images.size(0)
        train_loss = train_loss / len(train_loader.sampler)

        model.eval()
        with torch.no_grad():
            for images, _, _ in valid_loader:
                images = images.to(device)
                encoder_groups, decoder_groups = model(images)
                loss = global_cosine_hm(encoder_groups, decoder_groups, 0.0, config["shrink_factor"])
                valid_loss += loss.item() * images.size(0)
        valid_loss = valid_loss / len(valid_loader.sampler)

        writer.add_scalars("Loss", {"training": train_loss, "validation": valid_loss}, epoch)  # one plot, two curves
        print(f"Epoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}")

        if valid_loss <= valid_loss_min:
            print(f"Validation loss decreased ({valid_loss_min:.4f} --> {valid_loss:.4f}).  Saving model ...")
            torch.save({"bottleneck": model.bottleneck.state_dict(), "decoder": model.decoder.state_dict()}, save_path)
            valid_loss_min = valid_loss

    writer.close()


def main() -> None:
    """Index the dataset, train one model on all categories, checkpoint on validation improvement."""
    config = load_config()
    device = get_device()

    torch.manual_seed(17)                                          # fixed seeds: same validation split at every run
    np.random.seed(17)

    data_root = Path(config["data_root"])
    index_csv = data_root / "index.csv"
    num_indexed_images = index_mvtec(data_root, index_csv)
    print(f"index.csv: {num_indexed_images} images -> {index_csv}")

    train_loader, valid_loader, _, _ = build_dataloaders(
        index_csv, config["img_size"], config["crop_size"],
        config["batch_size"], config["valid_ratio"], config["num_workers"], apply_preprocessing,
    )
    print(
        f"{len(train_loader.sampler)} train images (normal only, all categories), "
        f"{len(valid_loader.sampler)} held out for validation"
    )

    model = build_model(config, device)
    num_frozen_parameters = sum(parameter.numel() for parameter in model.encoder.parameters())
    num_trainable_parameters = sum(
        parameter.numel() for module in (model.bottleneck, model.decoder) for parameter in module.parameters()
    )
    print(
        f"{config['backbone']} frozen ({num_frozen_parameters / 1e6:.0f}M parameters), "
        f"{num_trainable_parameters / 1e6:.1f}M trainable"
    )

    optimizer, scheduler = build_optimizer(model, config)
    save_path = checkpoint_path()
    train_dinomaly(model, train_loader, valid_loader, optimizer, scheduler, config, save_path)

    checkpoint = torch.load(save_path)                             # evaluate the best-validation weights, not the last epoch
    model.bottleneck.load_state_dict(checkpoint["bottleneck"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    threshold = compute_normal_threshold(model, valid_loader, config["top_pixel_ratio"])
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, save_path)
    print(f"Decision threshold (max validation score): {threshold:.4f}")
    print("Losses and first batch in TensorBoard: tensorboard --logdir runs")


if __name__ == "__main__":
    main()
