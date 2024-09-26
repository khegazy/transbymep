from .mlp import MLPpath
from .b_spline import BSpline
from .elastic_band import ElasticBand

path_dict = {
    "elastic_band" : ElasticBand,
    "mlp" : MLPpath,
    "bspline" : BSpline
}

def get_path(name, potential, initial_point, final_point, device='cuda', **config):
    print(config)
    name = name.lower()
    if name not in path_dict:
        raise ValueError(f"Cannot get path {name}, can only handle paths {path_dict.keys()}")

    path = path_dict[name](potential=potential, initial_point=initial_point, final_point=final_point, device=device, **config)
    
    return path 