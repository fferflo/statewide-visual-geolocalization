import jax
import einx
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import einx.nn.flax as einn
from .util import _import
from functools import partial
from typing import Callable
import types

Norm = partial(einn.Norm, "... [c]", epsilon=1e-5)

class SMD(nn.Module):
    @staticmethod
    def from_config(config):
        return SMD(
            channels=config["embedding-channels"],
        )

    channels: int
    heads: int = 8

    @nn.compact
    def __call__(self, x):
        x = Norm()(x)

        x = einn.Linear("... [c1->c2]", c2=self.channels // self.heads)(x)

        s = x.shape[1:-1]

        x = einn.Linear("b [s...->s2] c", s2=np.prod(s) * 4)(x)
        x = jax.nn.gelu(x)
        x = einn.Linear("b [s2->s...] c", s=s)(x)

        x = einn.Linear("b [s...->h] c", s=s, h=self.heads)(x)
        x = einx.rearrange("b h c -> b (h c)", x)

        return x

class SAFA(nn.Module):
    @staticmethod
    def from_config(config):
        return SAFA(
            channels=config["embedding-channels"],
            heads=config["heads"],
        )

    channels: int
    heads: int

    @nn.compact
    def __call__(self, x):
        x = Norm()(x)

        x = einn.Linear("... [c1->c2]", c2=self.channels // self.heads)(x)

        # Max pooling
        multiscale = len(x.shape) == 5
        if multiscale:
            k = x.shape[1]
            x = einx.rearrange("b k s... c -> (b k) s... c", x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        if multiscale:
            x = einx.rearrange("(b k) s... c -> b k s... c", x, k=k)

        s = x.shape[1:-1]

        w = einx.mean("... [c]", x)
        w = einn.Linear("b [s...->h s2]", h=self.heads, s2=np.prod(s) // 2)(w)
        w = einn.Linear("b [h s2->h s...]", s=s)(w)
        x = einx.dot("b s... c, b h s... -> b (h c)", x, w)

        return x

class SpatialSoftmax(nn.Module):
    @staticmethod
    def from_config(config):
        return SpatialSoftmax(
            channels=config["embedding-channels"],
            heads=config["heads"],
        )

    channels: int
    heads: int

    @nn.compact
    def __call__(self, x):
        x = Norm()(x)

        attn = einn.Linear("... [c1->h]", h=self.heads, bias=False)(x)
        attn = einx.softmax("b [...] h", attn)
        value = einn.Linear("... [c1->c2]", c2=self.channels)(x)
        x = einx.dot("b ... (h c), b ... h -> b (h c)", value, attn)
        return x

class Mean(nn.Module):
    @staticmethod
    def from_config(config):
        return Mean(channels=config["embedding-channels"])

    channels: int

    @nn.compact
    def __call__(self, x):
        x = einx.mean("b [...] c", x)
        x = Norm()(x)
        x = einn.Linear("... [c_in->c_out]", c_out=self.channels)(x)
        return x

class Sample4Geo(nn.Module):
    @staticmethod
    def from_config(config):
        return Sample4Geo(channels=config["embedding-channels"])

    channels: int

    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] == self.channels
        x = einx.mean("b [...] c", x)
        x = einn.Norm("... [c]", epsilon=1e-6)(x)
        return x

class FromVit(nn.Module):
    @staticmethod
    def from_config(config):
        return FromVit(channels=config["embedding-channels"])

    channels: int

    @nn.compact
    def __call__(self, x):
        x = Norm()(x)

        x0 = einn.Linear("... [c1->c2]", c2=self.channels)(x[..., 0, :])
        x1 = einn.Linear("... [c1->c2]", c2=self.channels)(x[..., 1, :])
        x = 0.5 * (x0 + x1)
        
        if len(x.shape) == 3:
            # Must have a single aerial image
            assert x.shape[1] == 1
            x = x[:, 0]

        return x

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])
def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

class Model(nn.Module):
    def from_config(config):
        if config["pv-aerial-shared-encoder"]:
            assert config["pv-encoder"] == config["aerial-encoder"]
            pv_encoder = aerial_encoder = _import(config["pv-encoder"])()
        else:
            pv_encoder = _import(config["pv-encoder"])()
            aerial_encoder = _import(config["aerial-encoder"])()

        return Model(
            scale_init=config["scale.init"],
            scale_learnable=config["scale.learnable"],
            pv_encoder=pv_encoder,
            aerial_encoder=aerial_encoder,
            pv_decoder=_import(config["pv-decoder"]).from_config(config),
            aerial_decoder=_import(config["aerial-decoder"]).from_config(config),
        )

    scale_init: float
    scale_learnable: bool
    pv_encoder: nn.Module
    aerial_encoder: nn.Module
    pv_decoder: nn.Module
    aerial_decoder: nn.Module

    @nn.compact
    def __call__(self, batch):
        print("Tracing...")
        if self.scale_learnable:
            scale = self.param("scale", shape=[], dtype="float32", init=np.log(self.scale_init))
            scale = jnp.exp(scale)
        else:
            scale = self.scale_init

        assert "pv" in vars(batch) or "aerial" in vars(batch)

        if "pv" in vars(batch):
            x = batch.pv.images
            x = preprocess(x)
            x = self.pv_encoder(x)
            x = self.pv_decoder(x)
            x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)
            x = x * scale
            pv_features = x
        else:
            pv_features = None

        if "aerial" in vars(batch):
            x = batch.aerial.images
            k = x.shape[1] # number of aerial images per cell
            x = einx.rearrange("b k s... c -> (b k) s... c", x)
            x = preprocess(x)
            x = self.aerial_encoder(x)
            x = einx.rearrange("(b k) s... c -> b k s... c", x, k=k)
            x = self.aerial_decoder(x)
            x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-5)
            x = x * scale
            aerial_features = x
        else:
            aerial_features = None

        metrics = {
            "scale": scale,
        }

        return types.SimpleNamespace(
            pv_features=pv_features,
            aerial_features=aerial_features,
        ), metrics
