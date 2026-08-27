"""Dinomaly architecture: frozen DINOv2 encoder, noisy bottleneck, linear-attention decoder.

Assembled by build_model. The model class is called ViTill in the official code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenViTEncoder(nn.Module):
    """Frozen ViT: collects the features of the target layers, no cls token, no registers."""

    def __init__(self, vit: nn.Module, target_layers: list) -> None:
        super().__init__()
        self.vit = vit
        self.target_layers = target_layers
        self.num_register_tokens = getattr(vit, "num_register_tokens", 0)

    def train(self, mode: bool = True):
        return super().train(False)  # always in eval: no dropout on the encoder side

    def forward(self, image: torch.Tensor) -> list:
        with torch.no_grad():
            tokens = self.vit.prepare_tokens_with_masks(image)
            layer_features = []
            for block_index, block in enumerate(self.vit.blocks):
                tokens = block(tokens)
                if block_index in self.target_layers:
                    layer_features.append(tokens[:, 1 + self.num_register_tokens:, :])
        return layer_features


class NoisyBottleneck(nn.Module):
    """Two-layer MLP whose internal Dropout plays the pseudo-anomaly role."""

    def __init__(self, embed_dim: int, drop_rate: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)


class LinearAttention(nn.Module):
    """LA(Q, K, V) = (elu(Q)+1) @ ((elu(K)+1)^T @ V), normalise par q . sum(K)."""

    def __init__(self, embed_dim: int, num_heads: int, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, embed_dim = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, num_tokens, 3, self.num_heads, embed_dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = F.elu(query) + 1.0
        key = F.elu(key) + 1.0
        key_value_context = torch.einsum("...sd,...se->...de", key, value)  # K^T V first, avoids the N^2 attention matrix
        normalizer = 1.0 / torch.einsum("...sd,...d->...s", query, key.sum(dim=-2))
        attention_output = torch.einsum("...de,...sd,...s->...se", key_value_context, query, normalizer)
        return self.output_projection(attention_output.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim))


class DecoderBlock(nn.Module):
    """Bloc pre-norm: attention melange les positions, MLP transforme chaque token."""

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-8)
        self.attn = LinearAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-8)
        feedforward_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim), nn.GELU(), nn.Linear(feedforward_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Dinomaly(nn.Module):
    """Frozen encoder -> bottleneck -> decoder. Outputs: (encoder, decoder) groups."""

    def __init__(self, encoder: nn.Module, bottleneck: nn.Module, decoder: nn.Module, groups: list) -> None:
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.groups = groups

    @staticmethod
    def fuse(layer_features: list) -> torch.Tensor:
        return torch.stack(layer_features, dim=1).mean(dim=1)

    def forward(self, image: torch.Tensor) -> tuple:
        encoder_layers = self.encoder(image)
        bottleneck_features = self.bottleneck(self.fuse(encoder_layers))
        decoder_layers = [block(bottleneck_features) for block in self.decoder]
        decoder_layers = decoder_layers[::-1]  # deep <-> shallow crossing (U-Net style, official-code detail)

        encoder_groups = [self.fuse([encoder_layers[layer_index] for layer_index in group]) for group in self.groups]
        decoder_groups = [self.fuse([decoder_layers[layer_index] for layer_index in group]) for group in self.groups]
        return encoder_groups, decoder_groups


def init_weights(module) -> None:
    """
    trunc_normal_ initialization for the trainable linear layers of the reconstruction
    branch (std 0.01 clamped to [-0.03, 0.03], biases to zero), as in the official code.

    Usage:
        model.bottleneck.apply(init_weights)
        model.decoder.apply(init_weights)
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model(config: dict, device: torch.device) -> Dinomaly:
    """Full pipeline: frozen DINOv2 backbone, bottleneck, 8-block decoder, weights initialized.

    Only the bottleneck and the decoder are trainable; the encoder is frozen and
    always in eval mode (no dropout on the encoder side).

    :param config: configuration dict (load_config).
    :param device: target device.
    :return: the assembled model, ready to train.
    """
    backbone = torch.hub.load(
        "facebookresearch/dinov2", config["backbone"], trust_repo=True, verbose=False
    ).to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    encoder = FrozenViTEncoder(backbone, config["target_layers"])
    bottleneck = NoisyBottleneck(config["embed_dim"], config["drop_rate"]).to(device)
    decoder = nn.ModuleList(
        [DecoderBlock(config["embed_dim"], num_heads=12) for _ in range(8)]  # paper: ViT-Base -> 12 heads, 8 blocks
    ).to(device)

    model = Dinomaly(encoder, bottleneck, decoder, config["groups"]).to(device)
    model.bottleneck.apply(init_weights)
    model.decoder.apply(init_weights)
    return model
