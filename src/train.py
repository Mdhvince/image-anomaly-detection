import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from config import PROJECT_ROOT, checkpoint_path, get_config, get_device
from dataset_utils import generate_dataframe_from_images, train_valid_split
from inference import compute_normal_threshold
from loss import build_optimizer, global_cosine_hm
from model import build_model
from preprocess import apply_preprocessing, denormalize
from src.custom_dataset import CustomDataset


def train_dinomaly(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader, optimizer: torch.optim.AdamW, scheduler: torch.optim.lr_scheduler.LambdaLR, config: dict, save_path: Path) -> None:
    writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "runs"))
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
                config["final_mining_percent"] * iteration / config["mining_ramp_iterations"],
                config["final_mining_percent"],
            )

            optimizer.zero_grad()
            encoder_groups, decoder_groups = model(images)
            loss = global_cosine_hm(encoder_groups, decoder_groups, mining_percent, config["shrink_factor"])
            loss.backward()
            nn.utils.clip_grad_norm_(                               # Clip the gradients (paper: max_norm 0.1)
                [
                    parameter
                    for module in (model.bottleneck, model.decoder)
                    for parameter in module.parameters()
                ],
                max_norm=config["max_grad_norm"],
            )
            optimizer.step()
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
    config = get_config()
    device = get_device()
    torch.manual_seed(17)
    np.random.seed(17)

    bs, ims, cs = config["batch_size"], config["img_size"], config["crop_size"]
    val_ratio = config["valid_ratio"]
    num_workers = config["num_workers"]

    data: pd.DataFrame = generate_dataframe_from_images(config["data_root"])
    train_data = data[data["split"] == "train"]
    train_set = CustomDataset(train_data, ims, cs, apply_preprocessing)
    train_sampler, valid_sampler = train_valid_split(train_set, val_ratio)
    train_loader = DataLoader(train_set, batch_size=bs, sampler=train_sampler, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(train_set, batch_size=bs, sampler=valid_sampler, num_workers=num_workers)

    model = build_model(config, device)
    optimizer, scheduler = build_optimizer(model, config)

    save_path = checkpoint_path()
    train_dinomaly(model, train_loader, valid_loader, optimizer, scheduler, config, save_path)

    checkpoint = torch.load(save_path)
    model.bottleneck.load_state_dict(checkpoint["bottleneck"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    threshold = compute_normal_threshold(model, valid_loader, config["top_pixel_ratio"])
    checkpoint["threshold"] = threshold
    torch.save(checkpoint, save_path)
    print(f"Decision threshold (max validation score): {threshold:.4f}")


if __name__ == "__main__":
    main()
