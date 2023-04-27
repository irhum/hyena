import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Any

Array = Any


class DecoderLayer(nn.Module):
    """A standard decoder layer with a residual connection."""

    features: int
    hidden_features: int
    mixer_fn: Callable[..., nn.Module]
    dropout: float = 0.0
    out_init: Callable[..., Array] = nn.linear.default_kernel_init

    @nn.compact
    def __call__(
        self,
        x,
        deterministic: bool = True,
        mode: str = "train",
        idxs: jnp.ndarray = None,
    ):
        residual = x
        x = nn.LayerNorm()(x)
        x = self.mixer_fn(
            features=self.features, out_init=self.out_init, dropout=self.dropout
        )(x, deterministic=deterministic, mode=mode, idxs=idxs)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = residual + x

        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(
            self.hidden_features, kernel_init=nn.initializers.normal(stddev=0.02)
        )(x)
        x = nn.gelu(x)
        x = nn.Dense(self.features, kernel_init=self.out_init)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = residual + x

        return x


class Decoder(nn.Module):
    """A decoder network composed of a stack of decoder layers."""

    embedding: nn.Module
    block_fn: Callable[..., nn.Module]
    num_layers: int
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        tokens,
        deterministic: bool = True,
        mode: str = "train",
        idxs: jnp.ndarray = None,
    ):
        embeds = self.embedding(tokens)
        embeds = nn.Dropout(rate=self.dropout)(embeds, deterministic=deterministic)

        for idx in range(self.num_layers):
            embeds = self.block_fn(name=f"{idx}", dropout=self.dropout)(
                embeds, deterministic=deterministic, mode=mode, idxs=idxs
            )

        embeds = nn.LayerNorm()(embeds)
        logits = self.embedding.attend(embeds)
        return logits * 1 / jnp.sqrt(self.embedding.features)
