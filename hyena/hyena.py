from functools import partial
from typing import Any, Callable, Sequence, Union, Tuple

import jax
import einops
import flax.linen as nn
import jax.numpy as jnp

Array = Any


class ExponentialModulation(nn.Module):
    """Applies exponential decay window to a batch of filtering signals.

    Ths ensures that, at initialization tokens primarily receive input from
    nearby, more recent tokens."""

    fast_decay_pct: float = 0.3
    slow_decay_pct: float = 1.5
    target: float = 1e-2
    shift: float = 0.0

    @nn.compact
    def __call__(self, t, h):
        # Get number of feature dimensions.
        features = h.shape[-1]
        # Compute the min and max decay coefficients.
        max_decay = jnp.log(self.target) / self.fast_decay_pct
        min_decay = jnp.log(self.target) / self.slow_decay_pct
        # Compute the decay coefficients per feature dimension.
        deltas = jnp.linspace(min_decay, max_decay, features)  # (D,)
        # Apply the decay window to each filter.
        decay = jnp.exp(-t * jnp.abs(deltas))  # (1, T, 1, D)
        h = h * (decay + self.shift)  # (1, T, 1, D)
        return h


class Siren(nn.Module):
    """This is a SIREN network used to generate the convolution filters.

    At initialization, the network is initialized to produce outputs of
    high frequency, which has been proven to lead to faster convergence
    when the target signals have a rich structure."""

    hidden_features: int
    out_features: Union[int, Sequence[int]]
    num_layers: int
    freq: float = 10.0

    @nn.compact
    def __call__(self, x):
        init_fn = partial(
            nn.initializers.variance_scaling, mode="fan_in", distribution="uniform"
        )
        # Dense is initialized with a uniform distribution
        # in [-1/sqrt(D_in), 1/sqrt(D_in)].
        x = nn.Dense(self.hidden_features, kernel_init=init_fn(1 / 3))(x)
        x = jnp.sin(self.freq * x)

        for _ in range(self.num_layers):
            # Dense is initialized with a uniform distribution
            # in [-6/(sqrt(D_h)), 6/(sqrt(D_h))].
            x = nn.Dense(self.hidden_features, kernel_init=init_fn(2.0))(x)
            x = jnp.sin(x)

        # Project to output dimension.
        x = nn.DenseGeneral(
            self.out_features, use_bias=False, kernel_init=init_fn(2.0 * 0.02)
        )(x)
        return x


class HyenaOperator(nn.Module):
    """This implements the Hyena operator, with an input and output projection."""

    features: int
    max_len: int
    filter_fn: Callable[[Tuple[int]], nn.Module]
    modulation_fn: Callable[[], nn.Module]
    order: int = 2
    dropout: float = 0.0
    out_init: nn.initializers.Initializer = nn.linear.default_kernel_init

    @nn.compact
    def __call__(
        self,
        u,
        deterministic: bool = True,
        mode: str = "train",
        idxs: jnp.ndarray = None,
    ):
        l = u.shape[-2]
        l_filter = min(l, self.max_len)

        if mode in ["train", "prefill"]:
            # Generate the filters.
            t = jnp.linspace(-1, 1, self.max_len)[None, :l_filter, None]  # (1, T, 1)
            h = self.filter_fn(out_features=(self.order, self.features))(
                t
            )  # (1, T, O, D)
            # Apply exponential decay window to filers.
            mod_t = jnp.linspace(0, 1, self.max_len)[
                None, :l_filter, None, None
            ]  # (1, T, 1, 1)
            h = self.modulation_fn()(mod_t, h)  # (1, T, O, D)
            # Reorder the filter axes for compatibility with input signal.
            h = einops.rearrange(h, "1 l o d -> o 1 l d", o=self.order)  # (O, 1, T, D)

        if mode in ["prefill", "decode"]:
            cached_h = self.variable(
                "cache",
                "h",
                jnp.zeros,
                (self.order, 1, l_filter, self.features),
                u.dtype,
            )
            if mode == "prefill":
                cached_h.value = h
            else:
                h = cached_h.value

        bias = self.param(
            "bias",
            nn.initializers.normal(stddev=1),
            (self.order, 1, 1, self.features),
        )  # (O, 1, 1, D)

        inner_width = self.features * (self.order + 2)
        # Affine projection "into" the layer
        u = nn.Dense(
            inner_width, name="in_proj", kernel_init=nn.initializers.normal(stddev=0.02)
        )(
            u
        )  # (B, T, (O+1)*D)

        if mode in ["prefill", "decode"]:
            cached_u = self.variable(
                "cache", "u", jnp.zeros, (u.shape[0], 3, u.shape[-1]), u.dtype
            )
            cached_idxs = self.variable(
                "cache", "idxs", jnp.zeros, (u.shape[0],), jnp.int32
            )
            if mode == "prefill":
                extract_fn = jax.vmap(
                    lambda x, idx: jax.lax.dynamic_slice(
                        x, (idx - 2, 0), (3, u.shape[-1])
                    ),
                    in_axes=0,
                    out_axes=0,
                )
                cached_u.value = extract_fn(u, idxs)
                cached_idxs.value = idxs
            else:
                u = cached_u.value.at[:, 0:1, :].set(u)
                u = jnp.roll(u, -1, axis=1)
                cached_u.value = u
                cached_idxs.value += 1
                idxs = cached_idxs.value

        # Note that we use a short "pre-filter" with window size 3.
        # Causal means only prior tokens are used to compute the next token,
        # with zero padding where needed.
        uc = nn.Conv(
            inner_width,
            kernel_size=(3,),
            padding="VALID" if mode == "decode" else "CAUSAL",
            feature_group_count=inner_width,
            name="short_filter",
        )(u)[
            :, :l_filter, :
        ]  # (B, T, (O+1)*D)

        # Get the generated "diagonals" and the input signal.
        *x, v = jnp.split(uc, self.order + 2, axis=-1)  # (B, T, D) * O, (B, T, D)
        v = v * x[0]  # (B, T, D)

        # We then apply the sequence of filters and diagonals
        for o, x_i in enumerate(x[1:]):
            if mode in ["prefill", "decode"]:
                cached_v = self.variable(
                    "cache",
                    f"v{o}",
                    jnp.zeros,
                    (v.shape[0], l_filter, v.shape[-1]),
                    v.dtype,
                )
                if mode == "prefill":
                    cached_v.value = v
                else:
                    cached_v.value = cached_v.value.at[
                        jnp.arange(0, v.shape[0]), idxs, :
                    ].set(v[:, 0, :])
                    v = cached_v.value

            if mode in ["train", "prefill"]:
                v = self.fftconv(v, h[o], bias[o])  # (B, T, D)
            else:
                v = self.inference_conv(v, idxs, h[o], bias[o])  # (B, T, D)
            v = v * x_i  # (B, T, D)

        v = nn.Dropout(rate=self.dropout, deterministic=deterministic)(v)
        # We then project back to the input space, to add as a residual.
        y = nn.Dense(self.features, name="out_proj", kernel_init=self.out_init)(
            v
        )  # (B, T, D)
        return y

    def fftconv(self, v, h, bias):
        # Zero pad to 2x the length to create causal convolution.
        seqlen = v.shape[-2]
        fft_size = 2 * seqlen

        # Real valued input signals, complex valued output frequencies.
        h_f = jnp.fft.rfft(h, n=fft_size, axis=-2)
        v_f = jnp.fft.rfft(v, n=fft_size, axis=-2)

        # Multiply in the frequency domain.
        y_f = v_f * h_f / fft_size
        # Invert FFT to get the output signal.
        y = jnp.fft.irfft(y_f, axis=-2, n=fft_size, norm="forward")[:, :seqlen, :]
        return y + v * bias

    def inference_conv(self, vs, idxs, h, bias):
        """Computes the post-convolution output of a *single* token.

        Args:
            vs: The input signals, shape (B, T, D).
            idxs: The token to compute the output for, shape (B,).
            h: The filters to convolve with.
            D: The bias term to add to the result.
        """

        def inner_fn(idx, v, h, bias):
            h = jnp.roll(h, idx + 1, axis=-2)
            mask = jnp.arange(h.shape[-2])[:, None] <= idx
            return (
                jnp.sum(v * h * mask, axis=-2, keepdims=True) + v[None, idx, :] * bias
            )

        h = h[0, ::-1, :]
        v_inner_fn = jax.vmap(inner_fn, in_axes=(0, 0, None, None), out_axes=0)
        return v_inner_fn(idxs, vs, h, bias[0, :, :])
