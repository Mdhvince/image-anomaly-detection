import math
from functools import partial

import torch
import torch.nn as nn


def modify_grad(gradient: torch.Tensor, easy_mask: torch.Tensor, shrink_factor: float) -> torch.Tensor:
    """sg(.)_shrink of the paper: gradient shrunk on the points already well reconstructed.

    :param gradient: incoming gradient of a decoder group (hook argument).
    :param easy_mask: True where the point is already well reconstructed.
    :param shrink_factor: multiplier applied on easy points (paper: 0.1).
    :return: the gradient with easy points shrunk.
    """
    return gradient * torch.where(easy_mask, shrink_factor, 1.0)


def global_cosine_hm(encoder_groups: list, decoder_groups: list, mining_percent: float, shrink_factor: float) -> torch.Tensor:
    """
    Global cosine loss per group; hard-mining hooks fire during loss.backward().

    :param encoder_groups: fused encoder features per group.
    :param decoder_groups: fused decoder features per group (gradient-carrying).
    :param mining_percent: fraction of easy points (lowest per-point cosine distance) whose gradient gets shrunk.
    :param shrink_factor: multiplier applied on easy points (paper: 0.1).
    :return: mean cosine loss over the groups.
    """
    loss = 0.0
    for encoder_group, decoder_group in zip(encoder_groups, decoder_groups):
        encoder_ref = encoder_group.detach()

        with torch.no_grad():
            # per-point cosine distance: over the channels (dim=-1), as in the official code
            point_distances = 1 - nn.functional.cosine_similarity(encoder_ref, decoder_group, dim=-1)  # (B, N)
            num_hard_points = int(point_distances.numel() * (1 - mining_percent))
            distance_threshold = torch.topk(point_distances.flatten(), num_hard_points)[0][-1]
            easy_points_mask = (point_distances < distance_threshold).unsqueeze(-1)  # (B, N, 1), broadcast over the channels

        loss = loss + (1 - nn.functional.cosine_similarity(encoder_ref.flatten(1), decoder_group.flatten(1))).mean()
        if torch.is_grad_enabled():                                # the hook only shapes gradients: useless outside a backward
            decoder_group.register_hook(partial(modify_grad, easy_mask=easy_points_mask, shrink_factor=shrink_factor))
    return loss / len(encoder_groups)


def compute_lr_ratio(iteration: int, warmup_iterations: int, num_iterations: int, final_lr_ratio: float) -> float:
    """Linear warmup then cosine decay, as a ratio of the initial learning rate.

    :param iteration: current iteration (scheduler argument).
    :param warmup_iterations: iterations of linear ramp-up.
    :param num_iterations: total training iterations.
    :param final_lr_ratio: lr_final / lr_init reached at the last iteration.
    :return: learning-rate multiplier for the iteration.
    """
    if iteration < warmup_iterations:
        return iteration / max(warmup_iterations, 1)
    progress = (iteration - warmup_iterations) / max(num_iterations - warmup_iterations, 1)
    return final_lr_ratio + (1 - final_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def build_optimizer(model: torch.nn.Module, config: dict) -> tuple:
    """AdamW(amsgrad) + warmup/cosine schedule on the trainable modules (bottleneck + decoder).

    :param model: assembled Dinomaly model.
    :param config: configuration dict (get_config).
    :return: (optimizer, scheduler).
    """
    trainable_modules = nn.ModuleList([model.bottleneck, model.decoder])
    optimizer = torch.optim.AdamW(
        params=trainable_modules.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=config["weight_decay"],
        amsgrad=True,
        eps=1e-10,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            compute_lr_ratio,
            warmup_iterations=config["warmup_iterations"],
            num_iterations=config["num_iterations"],
            final_lr_ratio=config["final_learning_rate"] / config["learning_rate"],
        ),
    )
    return optimizer, scheduler
