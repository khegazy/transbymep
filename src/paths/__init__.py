import jax.tree_util as jtu
import equinox as eqx

from .mlp import MLPpath
from .mlp_dist import MLPdistpath
from .mlp_invdist import MLPinvdistpath
from .mlp_expdist import MLPexpdistpath
from .b_spline import BSpline
from .elastic_band import ElasticBand

path_dict = {
    "elastic_band" : ElasticBand,
    "mlp" : MLPpath,
    "bspline" : BSpline,
    "mlpdist" : MLPdistpath,
    "mlpinvdist" : MLPinvdistpath,
    "mlpexpdist" : MLPexpdistpath,
}

def get_path(name, potential, initial_point, final_point, **config):
    print(config)
    name = name.lower()
    if name not in path_dict:
        raise ValueError(f"Cannot get path {name}, can only handle paths {path_dict.keys()}")

    path = path_dict[name](potential, initial_point, final_point, **config)
    
    return path 