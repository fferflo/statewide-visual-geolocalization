import jax
import types

def flatten(simple_ns):
    children = []
    children_names = []
    for name, value in sorted(vars(simple_ns).items()):
        children.append(value)
        children_names.append(name)

    return children, children_names

def unflatten(aux_data, children):
    children_names = aux_data
    d = {name: value for name, value in zip(children_names, children)}
    return types.SimpleNamespace(**d)

jax.tree_util.register_pytree_node(types.SimpleNamespace, flatten, unflatten)

from .model import *
from .util import *