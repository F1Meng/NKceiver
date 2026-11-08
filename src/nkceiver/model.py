from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class NKceiverOutput:
    embedding: Tensor
    leaf_logits: Tensor
    token_memory: Tensor


class StateEncoder(nn.Module):
    def __init__(self, n_states, expression_mean, expression_std, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.register_buffer("expression_mean", expression_mean.float())
        self.register_buffer("expression_std", expression_std.float().clamp_min(0.05))
        self.expression_encoder = nn.Sequential(
            nn.Linear(expression_mean.numel(), 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )
        # Preserve the frozen training seed's RNG sequence without adding a learned feature.
        torch.empty(n_states + 1, hidden_dim).normal_()
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, state_expression, state_present, state_proportion=None):
        x = (state_expression - self.expression_mean) / self.expression_std
        tokens = self.output_norm(self.expression_encoder(x))
        return torch.where(state_present.unsqueeze(-1), tokens, torch.zeros_like(tokens))


class NKceiver(nn.Module):
    def __init__(
        self, n_states, expression_mean, expression_std, n_classes=18, hidden_dim=128, dropout=0.1
    ):
        super().__init__()
        self.tokenizer = StateEncoder(
            n_states, expression_mean, expression_std, hidden_dim, dropout
        )
        self.latents = nn.Parameter(torch.randn(8, hidden_dim) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, 4, dropout=dropout, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            hidden_dim, 4, hidden_dim * 2, dropout, "gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(hidden_dim))
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.disease_head = nn.Linear(hidden_dim, n_classes)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.cross_ffn_norm = nn.LayerNorm(hidden_dim)
        self.cross_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.cross_dropout = nn.Dropout(dropout)

    def encode(self, **inputs: Tensor) -> tuple[Tensor, Tensor]:
        tokens = self.tokenizer(**inputs)
        queries = self.latents.unsqueeze(0).expand(tokens.shape[0], -1, -1)
        valid = inputs["state_present"].bool()
        if not valid.any(1).all():
            raise ValueError(
                "No state has sufficient cell support; abstain instead of generating a prediction"
            )
        padding = ~valid
        context = self.context_norm(tokens)
        delta, _ = self.cross_attention(
            self.query_norm(queries), context, context, key_padding_mask=padding, need_weights=False
        )
        memory = queries + self.cross_dropout(delta)
        memory = memory + self.cross_dropout(self.cross_ffn(self.cross_ffn_norm(memory)))
        memory = self.encoder(memory)
        pooled = memory.mean(1)
        phenotype = self.projection(pooled)
        return (phenotype, memory)

    def forward(self, **inputs: Tensor) -> NKceiverOutput:
        embedding, memory = self.encode(**inputs)
        return NKceiverOutput(embedding, self.disease_head(embedding), memory)


def build_model(config, mean, std, classes, seed):
    if config["architecture"] != "residual":
        raise ValueError("Expected architecture: residual")
    if config.get("model", "perceiver") != "perceiver":
        raise ValueError("Expected model: perceiver")
    if config["feature"] != "M" or config["presence_mode"] != "validity_mask":
        raise ValueError("Expected feature M and presence_mode validity_mask")
    if config.get("state_identity", False) or not config.get("attention_mask", True):
        raise ValueError("Expected state_identity false and attention_mask true")
    torch.manual_seed(seed)
    model = NKceiver(
        n_states=config.get("n_states", 12),
        expression_mean=mean,
        expression_std=std,
        n_classes=classes,
        hidden_dim=128,
        dropout=config["dropout"],
    )
    return model
