import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .mlp import MLPpath
from .b_spline import BSpline
from .elastic_band import ElasticBand

path_dict = {
    "elastic_band" : ElasticBand,
    "mlp" : MLPpath,
    "bspline" : BSpline
}

def get_path(name, potential, initial_point, final_point, process, **config):
    name = name.lower()
    if name not in path_dict:
        raise ValueError(f"Cannot get path {name}, can only handle paths {path_dict.keys()}")

    path = path_dict[name](potential, initial_point, final_point, **config)
    if process.is_distributed:
        print("DEVICE ID", process.rank, process.device_ids)
        #torch.cpu.set_device(process.local_rank)
        if process.device_type == 'cpu':
            path = DDP(path)
        else:
            path = DDP(path, device_ids=process.device_ids, output_device=process.local_rank)
        path.get_path = path.module.get_path
        print("DEVICE", path.module.layers[0].weight.get_device())
    
    return path 