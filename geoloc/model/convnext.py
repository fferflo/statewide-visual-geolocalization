from flax import linen as nn
import einx.nn.flax as einn
from functools import partial
import numpy as np
import einx
import transformers
import weightbridge

Linear = partial(einn.Linear, "... [_->features]")
Norm = partial(einn.Norm, "... [c]", epsilon=1e-6)

class Stem(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.channels, kernel_size=(4, 4), strides=(4, 4), padding=0)(x)
        x = Norm()(x)
        return x

class Downsample(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x = Norm()(x)
        x = nn.Conv(features=self.channels, kernel_size=(2, 2), strides=(2, 2), padding=0)(x)
        return x

class Block(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        x0 = x

        x = nn.Conv(features=self.channels, kernel_size=(7, 7), strides=(1, 1), feature_group_count=x.shape[-1], padding=3)(x)
        x = Norm()(x)
        x = Linear(features=self.channels * 4)(x)
        x = nn.gelu(x)
        x = Linear(features=self.channels)(x)

        x = einx.multiply("... [c]", x, self.param)
        x = x0 + x

        return x

class Stage(nn.Module):
    depth: int
    channels: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            x = Block(self.channels)(x)

        return x

class ConvNeXt(nn.Module):
    @staticmethod
    def tiny():
        return ConvNeXt(
            depths=[3, 3, 9, 3],
            channels=[96, 192, 384, 768],
        )
    
    @staticmethod
    def base():
        return ConvNeXt(
            depths=[3, 3, 27, 3],
            channels=[128, 256, 512, 1024],
        )

    @staticmethod
    def base_adaptweights(params):
        pretrained_params = {k: np.asarray(v) for k, v in transformers.ConvNextModel.from_pretrained("facebook/convnext-base-384-22k-1k").state_dict().items() if not k.startswith("layernorm")}
        return weightbridge.adapt(pretrained_params, params, in_format="pytorch", out_format="flax")

    depths: list[int]
    channels: list[int]

    @nn.compact
    def __call__(self, x):
        x = Stem(self.channels[0])(x)

        for stage_index in range(len(self.depths)):
            if stage_index > 0:
                x = Downsample(self.channels[stage_index])(x)

            x = Stage(self.depths[stage_index], self.channels[stage_index])(x)

        return x

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])
def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color
