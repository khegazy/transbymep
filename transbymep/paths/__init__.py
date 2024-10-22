from .mlp import MLPpath
from .mlp_dist import MLPDISTpath
from .b_spline import BSpline
from .elastic_band import ElasticBand
from .bezier import Bezier

path_dict = {
    "elastic_band" : ElasticBand,
    "mlp" : MLPpath,
    "mlp_dist" : MLPDISTpath,
    "bspline" : BSpline,
    "bezier" : Bezier,
}

def get_path(name, potential, initial_point, final_point, device='cuda', **config):
    print(config)
    name = name.lower()
    path = path_dict[name](potential=potential, initial_point=initial_point, final_point=final_point, device=device, **config)
    
    return path 