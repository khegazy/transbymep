import jax.tree_util as jtu
import equinox as eqx

from .mlp import MLPpath
from .b_spline import BSpline
from .elastic_band import ElasticBand

path_dict = {
    "elastic_band" : ElasticBand,
    "mlp" : MLPpath,
    "bspline" : BSpline
}

def get_path(name, potential, initial_point, final_point, **config):
    print(config)
    name = name.lower()
    if name not in path_dict:
        raise ValueError(f"Cannot get path {name}, can only handle paths {path_dict.keys()}")

    path = path_dict[name](potential, initial_point, final_point, **config)
    
    filter_spec = jtu.tree_map(lambda _: True, path)
    filter_spec = eqx.tree_at(
        path.tree_filter_fxn,
        #lambda tree: (tree.initial_point, tree.final_point, tree.potential),
        filter_spec,
        replace=[False,]*path.tree_filter_fxn(None, get_len=True),
        #replace=(False, False, False),
    )
    return path, filter_spec