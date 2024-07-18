import traceback, jax
import jax.numpy as jnp
from jax.experimental import checkify
import importlib

def check_finite(x):
    message = f"\n##########BEGIN##########\n" + "".join(traceback.format_stack()[-10:]) + "\n##########END##########"
    leaves, _ = jax.tree_util.tree_flatten(x)
    if len(leaves) > 0:
        leaves = jnp.concatenate([l.reshape([-1]) for l in leaves], axis=0)
        checkify.check(jnp.all(jnp.isfinite(leaves)), f"Got non-finite: {message}")
    return x

def check_nan(x):
    message = f"\n##########BEGIN##########\n" + "".join(traceback.format_stack()[-10:]) + "\n##########END##########"
    leaves, _ = jax.tree_util.tree_flatten(x)
    if len(leaves) > 0:
        leaves = jnp.concatenate([l.reshape([-1]) for l in leaves], axis=0)
        checkify.check(jnp.all(jnp.logical_not(jnp.isnan(leaves))), f"Got nan: {message}")
    return x

def _import(s):
    parts = s.split(".")

    if len(parts) == 1:
        if s in globals()["__builtins__"]:
            return globals()["__builtins__"][s]
    else:
        for n in range(len(parts)):
            module_parts = parts[:len(parts) - n]
            func_parts = parts[len(parts) - n:]
            try:
                x = importlib.import_module(".".join(module_parts))
                for part in func_parts:
                    x = getattr(x, part)
                return x
            except ModuleNotFoundError as e:
                pass
    raise ParseException(f"{s} could not be imported")