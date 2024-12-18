import torch
from torch import optim
from torch.optim import lr_scheduler
from transbymep.tools import scheduler

from transbymep.tools import Metrics

optimizer_dict = {
    "sgd" : optim.SGD,
    "adagrad" : optim.Adagrad,
    "adam" : optim.Adam,
}
scheduler_dict = {
    "step" : lr_scheduler.StepLR,
    "linear" : lr_scheduler.LinearLR,
    "multi_step" : lr_scheduler.MultiStepLR,
    "exponential" : lr_scheduler.ExponentialLR,
    "cosine" : lr_scheduler.CosineAnnealingLR,
    "cosine_restart" : lr_scheduler.CosineAnnealingWarmRestarts,
    "reduce_on_plateau" : lr_scheduler.ReduceLROnPlateau,
}
loss_scheduler_dict = {
    "linear" : scheduler.Linear,
    "cosine" : scheduler.Cosine,
    "reduce_on_plateau" : scheduler.ReduceOnPlateau,
    "increase_on_plateau" : scheduler.IncreaseOnPlateau,
}


class PathOptimizer(Metrics):
    def __init__(
            self,
            name,
            path,
            path_loss_names=None,
            path_loss_scales=torch.ones(1),
            TS_time_loss_names=None,
            TS_time_loss_scales=torch.ones(1),
            TS_region_loss_names=None,
            TS_region_loss_scales=torch.ones(1),
            device='cpu',
            **config
        ):
        super().__init__()
        
        # Initialize loss information
        self.path_loss_name = path_loss_names
        self.path_loss_scales = path_loss_scales
        self.TS_time_loss_names = TS_time_loss_names
        self.TS_time_loss_scales = TS_time_loss_scales
        self.TS_time_loss_fxn, _ = self.get_ode_eval_fxn(
            True, self.TS_time_loss_names, self.TS_time_loss_scales
        )
        self.TS_region_loss_names = TS_region_loss_names
        self.TS_region_loss_scales = TS_region_loss_scales
        self.TS_region_loss_fxn, _ = self.get_ode_eval_fxn(
            True, self.TS_region_loss_names, self.TS_region_loss_scales
        )
        self.has_TS_loss = self.TS_time_loss_names is not None\
            or self.TS_region_loss_names is not None
        self.TS_time = torch.empty(0)
        self.TS_region = torch.empty(0)
        self.device=device
        
        # Initialize optimizer
        name = name.lower()
        self.optimizer = optimizer_dict[name](path.parameters(), **config)
        self.scheduler = None
        self.loss_scheduler = None
        self.converged = False

    def set_scheduler(self, name, **config):
        name = name.lower()
        if name not in scheduler_dict:
            raise ValueError(f"Cannot handle scheduler type {name}, either add it to scheduler_dict or use {list(scheduler_dict.keys())}")
        self.scheduler = scheduler_dict[name](self.optimizer, **config)

    def set_loss_scheduler(self, **kwargs):
        self.loss_scheduler = {}
        for key, value in kwargs.items():
            name = value.pop('name').lower()
            if name not in loss_scheduler_dict:
                raise ValueError(f"Cannot handle loss scheduler type {name}, either add it to loss_scheduler_dict or use {list(loss_scheduler_dict.keys())}")
            if name == "reduce_on_plateau" or name == "increase_on_plateau":
                self.loss_scheduler[key] = loss_scheduler_dict[name](lr_scheduler=self.scheduler, **value)
            else:
                self.loss_scheduler[key] = loss_scheduler_dict[name](**value)
    
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
            path, self.path_loss_name, self.path_loss_scales, t_init=t_init, t_final=t_final
        )
        integral_loss = path_integral.integral
        #for n, prm in path.named_parameters():
        #    print(n, prm.grad)
        #print("path integral", path_integral)
        if integrator._integrator.max_batch is None:
            #TODO: Better to change this to backprop if not detached
            integral_loss.backward()
            # (path_integral.integral**2).backward()
        
        #############  Testing TS Loss ############
        # Evaluate TS loss functions
        if self.has_TS_loss and len(self.TS_time) > 0:
            TS_loss = torch.zeros(1)
            if self.TS_time_loss_fxn is not None:
                TS_time_loss = self.TS_time_loss_fxn(
                    self.TS_time, path
                )
                TS_loss = TS_loss + TS_time_loss 
            if self.TS_region_loss_fxn is not None:
                TS_region_loss = self.TS_region_loss_fxn(
                    self.TS_region, path
                )
                TS_loss = TS_loss + TS_region_loss 
            TS_loss.backward()        
        ###########################################

        self.optimizer.step()
        if self.scheduler is not None:
            if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(path_integral.loss)
                # time = path_integral.t.flatten()
                # time = time[len(time)//10:-len(time)//10]
                # force = path(time, return_force=True).path_force
                # self.scheduler.step(torch.linalg.norm(force, dim=-1).min())
                if self.optimizer.param_groups[0]['lr'] <= self.scheduler.min_lrs[0]:
                    self.converged = True
            else:
                self.scheduler.step()
        
        ############# Testing ##############
        # Find transition state time
        self.TS_range = torch.linspace(0.45, 0.65, 25)
        self.TS_time = torch.tensor([0.5])
        ####################################

        if self.loss_scheduler is not None:
            for key, loss_scheduler in self.loss_scheduler.items():
                loss_scheduler.step()
            metric_parameters = {key: loss_scheduler.get_value() for key, loss_scheduler in self.loss_scheduler.items()}
            integrator.update_metric_parameters(metric_parameters)
        return path_integral

    