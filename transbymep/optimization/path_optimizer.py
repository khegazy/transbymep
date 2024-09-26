import os
import yaml
import torch
from torch import optim
from torch.optim import lr_scheduler

optimizer_dict = {
    "sgd" : optim.SGD,
    "adagrad" : optim.Adagrad,
    "adam" : optim.Adam
}
scheduler_dict = {
    "step" : lr_scheduler.StepLR,
    "multi_step" : lr_scheduler.MultiStepLR,
    "exponential" : lr_scheduler.ExponentialLR,
    "cosine" : lr_scheduler.CosineAnnealingLR,
    "plateau" : lr_scheduler.ReduceLROnPlateau
}

class PathOptimizer():
    def __init__(
            self,
            name,
            # config,
            path,
            loss_name,
            # config_path=None,
            # path_type=None,
            # potential_type=None,
            # config_tag="",
            # config_dir="./optimizations/configs/",
            # expect_config=False,
            device='cpu',
            **config
        ):
        self.loss_name = loss_name
        # name = name.lower()
        self.device=device
        # if name not in optimizer_dict:
        #     raise ValueError(f"Cannot handle optimizer type {name}, either add it to optimizer_dict or use {list(optimizer_dict.keys())}")

        # if config_path is None and (path_type is None and potential_type is None):
        #     raise ValueError(f"get_optimizer requires either config_path")
        
        # Import saved optimizer config and combine with input config
        # config_path_vars = path_type is not None and potential_type is not None 
        # if config_path is None and not config_path_vars and not expect_config:
        #     print("Skipping optimizer config import")
        # elif config_path is not None or config_path_vars:
        #     config.update(
        #         self._import_optimizer_config(
        #             name,
        #             path_type,
        #             potential_type,
        #             tag=config_tag,
        #             dir=config_dir,
        #             is_expected=expect_config
        #         )
        #     )
        # else:
        #     raise ValueError("get_optimizer requires either config_path or both path_type and potential_type to be specified to import the config file.")
        
        # Initialize optimizer
        self.optimizer = optimizer_dict[name](path.parameters(), **config)
        self.scheduler = None
        self.converged = False

    def set_scheduler(self, name, **config):
        name = name.lower()
        if name not in scheduler_dict:
            raise ValueError(f"Cannot handle scheduler type {name}, either add it to scheduler_dict or use {list(scheduler_dict.keys())}")
        self.scheduler = scheduler_dict[name](self.optimizer, **config)
    
    # def _import_optimizer_config(
    #         self,
    #         name, 
    #         path_type,
    #         potential_type,
    #         tag="",
    #         dir="./optimizations/configs/",
    #         is_expected=True
    #     ):
    #     filename = f"{name}_{potential_type}_{path_type}"
    #     filename += f"_{tag}.yaml" if tag != "" else ".yaml"
    #     address = os.path.join(dir, filename)
    #     print(f"Importing optimizer config {address}")
    #     if os.path.exists(address):
    #         with open(address, 'r') as file:
    #             loaded_yaml = yaml.safe_load(file)
    #         return loaded_yaml
    #     elif is_expected:
    #         raise ImportError(f"Cannot find required file {address}")
    #     else:
    #         ImportWarning(f"Cannot find file {address}, still running")
    #         return {}
    
    def optimization_step(
            self,
            path,
            integrator,
            t_init=torch.tensor([0.], dtype=torch.float64),
            t_final=torch.tensor([1.], dtype=torch.float64)
        ):
        t_init = t_init.to(torch.float64).to(self.device)
        t_final = t_final.to(torch.float64).to(self.device)
        self.optimizer.zero_grad()
        path_integral = integrator.path_integral(
            path, self.loss_name, t_init=t_init, t_final=t_final
        )
        #for n, prm in path.named_parameters():
        #    print(n, prm.grad)
        #print("path integral", path_integral)
        if integrator._integrator.max_batch is None:
            #TODO: Better to change this to backprop if not detached
            path_integral.integral.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(path_integral.loss)
                if self.optimizer.param_groups[0]['lr'] <= self.scheduler.min_lrs[0]:
                    self.converged = True
            else:
                self.scheduler.step()
        return path_integral

    