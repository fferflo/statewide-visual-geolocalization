import optax
import jax.numpy as jnp
import numpy as np
import jax.tree_util

def from_config(params, config):
    def schedule(step):
        train_batchsize = config["train.batchsize"]

        if "warmup-steps" in config["train.schedule"]:
            warmup_steps = config["train.schedule.warmup-steps"]
        else:
            warmup_steps = 0

        type = config["train.schedule.type"]
        if type == "constant":
            warmup_factor = step / max(warmup_steps, 1)
            s = jnp.where(step < warmup_steps, warmup_factor, 1.0)
        elif type == "cosine":
            total_steps = config["train.schedule.total-steps"] - warmup_steps
            alpha = config["train.schedule.alpha"]
            cos_factor = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step - warmup_steps, total_steps) / total_steps)) * (1.0 - alpha) + alpha
            warmup_factor = step / max(warmup_steps, 1)
            s = jnp.where(step < warmup_steps, warmup_factor, cos_factor)
        else:
            assert False

        return s

    transforms = []
    if "grad-clip-norm" in config["train"]:
        transforms.append(optax.clip_by_global_norm(config["train.grad-clip-norm"]))
    if "grad-clip-value" in config["train"]:
        transforms.append(optax.clip(config["train.grad-clip-value"]))
    transforms.append(optax.scale_by_adam())
    transforms.append(optax.scale(config["train.learning-rate"]))
    if "weight-decay" in config["train"]:
        def decay(key, x):
            key = "/".join([k.key for k in key])
            return any(key.endswith(n) for n in ["/kernel", "/weight"])
        transforms.append(optax.add_decayed_weights(
            weight_decay=config["train.weight-decay"],
            mask=jax.tree_util.tree_map_with_path(decay, params),
        ))
    transforms.append(optax.scale_by_schedule(schedule))
    transforms.append(optax.scale(-1.0))
    optimizer = optax.chain(*transforms)

    return optimizer, schedule