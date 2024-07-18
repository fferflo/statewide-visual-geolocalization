import jax
import numpy as np

def pad(y, n):
    p = None
    def pad1(x):
        nonlocal p, y
        p2 = n - x.shape[0]
        if p is None:
            p = p2
        else:
            assert p == p2, f"{p} != {p2}"
        if p > 0:
            return np.concatenate([
                x,
                np.repeat(x[:1], axis=0, repeats=p),
            ], axis=0)
        else:
            return x
    y = jax.tree.map(pad1, y)
    return y, p

def unpad(x, p):
    def unpad1(x):
        return x[:-p] if p > 0 else x
    return jax.tree.map(unpad1, x)
