import configparser
import os
from pathlib import Path

import torch


CONFIG_ENV_VAR = "DINOMALITY_CONFIG"


def get_device() -> torch.device:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return torch.device(device)


def parse_integer_list(raw_value: str) -> list:
    """
    Parse a comma-separated ini value into a list of ints.
    :param raw_value: "2,3,4,5".
    :return: [2, 3, 4, 5].
    """
    return [int(token) for token in raw_value.split(",")]


def parse_group_list(raw_value: str) -> list:
    """
    Parse a semicolon-separated list of comma-separated integer lists.
    :param raw_value: "0,1,2,3;4,5,6,7".
    :return: [[0, 1, 2, 3], [4, 5, 6, 7]].
    """
    return [parse_integer_list(group) for group in raw_value.split(";")]


def load_config(config_path: Path | None = None) -> dict:
    """Read config.ini (or the file pointed by DINOMALITY_CONFIG) into the config dict.

    :param config_path: explicit ini file; default = DINOMALITY_CONFIG env or ./config.ini.
    :return: the configuration dict shared by train.py and inference.py.
    :raises FileNotFoundError: if the ini file does not exist.
    """
    path = Path(config_path or os.environ.get(CONFIG_ENV_VAR, "config.ini"))
    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    if not parser.read(path):
        raise FileNotFoundError(f"config file not found: {path.resolve()}")

    dataset = parser["dataset"]
    model = parser["model"]
    preprocessing = parser["preprocessing"]
    training = parser["training"]
    evaluation = parser["evaluation"]

    config = {
        "backbone": model["backbone"],
        "target_layers": parse_integer_list(model["target_layers"]),
        "embed_dim": model.getint("embed_dim"),
        "drop_rate": model.getfloat("drop_rate"),
        "groups": parse_group_list(model["groups"]),
        "img_size": preprocessing.getint("img_size"),
        "crop_size": preprocessing.getint("crop_size"),
        "batch_size": training.getint("batch_size"),
        "num_workers": training.getint("num_workers"),
        "valid_ratio": training.getfloat("valid_ratio"),
        "num_iterations": training.getint("num_iterations"),
        "lr": training.getfloat("lr"),
        "final_lr": training.getfloat("final_lr"),
        "weight_decay": training.getfloat("weight_decay"),
        "warmup_iterations": training.getint("warmup_iterations"),
        "max_grad_norm": training.getfloat("max_grad_norm"),
        "final_mining_percent": training.getfloat("final_mining_percent"),
        "mining_ramp_iters": training.getint("mining_ramp_iters"),
        "shrink_factor": training.getfloat("shrink_factor"),
        "top_pixel_ratio": evaluation.getfloat("top_pixel_ratio"),
        "data_root": dataset["data_root"].strip(),
    }

    # DEMO: lighter regimen, same mechanics. enabled = false for the paper regimen.
    demo = parser["demo"]
    if demo.getboolean("enabled"):
        config.update({
            "img_size": demo.getint("img_size"),
            "crop_size": demo.getint("crop_size"),
            "batch_size": demo.getint("batch_size"),
            "num_iterations": demo.getint("num_iterations"),
            "mining_ramp_iters": min(config["mining_ramp_iters"], max(1, demo.getint("num_iterations") // 10)),
        })
    return config


def checkpoint_path() -> Path:
    """Checkpoint file of the single multi-class model (bottleneck + decoder weights), parent created.

    :return: path of the checkpoint file.
    """
    path = Path("checkpoints") / "dinomaly.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
