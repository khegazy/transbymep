import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.functional import interpolate
from transbymep.tools import scheduler
from transbymep.tools.scheduler import get_schedulers

from transbymep.tools import Metrics

OPTIMIZER_DICT = {
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


class PathOptimizer():
    def __init__(
            self,
            path,
            path_loss_schedulers=None,
            path_ode_schedulers=None,
            TS_time_loss_names=None,
            TS_time_loss_scales=torch.ones(1),
            TS_time_loss_schedulers=None,
            TS_region_loss_names=None,
            TS_region_loss_scales=torch.ones(1),
            TS_region_loss_schedulers=None,
            device='cpu',
            **config
        ):
        super().__init__()
        
        self.device=device
        self.iteration = 0
        
        ####  Initialize loss information  #####
        self.has_TS_loss = TS_time_loss_names is not None\
            or TS_region_loss_names is not None
        
        self.TS_time_loss_names = TS_time_loss_names
        self.TS_time_loss_scales = TS_time_loss_scales
        self.TS_time_metrics = Metrics()
        self.TS_time_metrics.create_ode_fxn(
            True, self.TS_time_loss_names, self.TS_time_loss_scales
        )
        
        self.TS_region_loss_names = TS_region_loss_names
        self.TS_region_loss_scales = TS_region_loss_scales
        self.TS_region_metrics = Metrics()
        self.TS_region_metrics.create_ode_fxn(
            True, self.TS_region_loss_names, self.TS_region_loss_scales
        )
        
        #####  Initialize schedulers  #####
        self.ode_fxn_schedulers = get_schedulers(path_ode_schedulers)
        self.path_loss_schedulers = get_schedulers(path_loss_schedulers)
        self.TS_time_loss_schedulers = get_schedulers(TS_time_loss_schedulers)
        self.TS_region_loss_schedulers = get_schedulers(TS_region_loss_schedulers)
        

        #####  Initialize optimizer  #####
        #name = name.lower()
        assert 'optimizer' in config, "Must specify optimizer parameters (dict) with key 'optimizer"
        assert 'name' in config['optimizer'], f"Must specify name of optimizer: {list(OPTIMIZER_DICT.keys())}"
        opt_name = config['optimizer']['name'].lower()
        del config['optimizer']['name']
        self.optimizer = OPTIMIZER_DICT[opt_name](path.parameters(), **config['optimizer'])
        self.scheduler = None
        self.loss_scheduler = None
        self.converged = False

    """
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
    """
    
    def optimization_step(
            self,
            path,
            integrator,
            t_init=torch.tensor([0.], dtype=torch.float64),
            t_final=torch.tensor([1.], dtype=torch.float64)
        ):
        self.optimizer.zero_grad()
        t_init = t_init.to(torch.float64).to(self.device)
        t_final = t_final.to(torch.float64).to(self.device)
        ode_fxn_scales = {
            name : schd.get_value() for name, schd in self.ode_fxn_schedulers.items()
        }
        path_loss_scales = {
            name : schd.get_value() for name, schd in self.path_loss_schedulers.items()
        }
        path_loss_scales['iteration'] = self.iteration,
        TS_time_loss_scales = {
            name : schd.get_value() for name, schd in self.TS_time_loss_schedulers.items()
        }
        TS_region_loss_scales = {
            name : schd.get_value() for name, schd in self.TS_region_loss_schedulers.items()
        }
        path_integral = integrator.path_integral(
            path, #self.path_loss_name, self.path_loss_scales,
            ode_fxn_scales=ode_fxn_scales,
            loss_scales=path_loss_scales,
            t_init=t_init,
            t_final=t_final
        )
        #for n, prm in path.named_parameters():
        #    print(n, prm.grad)
        #print("path integral", path_integral)
        if not path_integral.gradient_taken:
            path_integral.loss.backward()
            # (path_integral.integral**2).backward()
        
        #############  Testing TS Loss ############
        # Evaluate TS loss functions
        if self.has_TS_loss and path.TS_time is not None:
            TS_loss = torch.zeros(1)
            if self.TS_time_metrics.ode_fxn is not None:
                self.TS_time_metrics.update_ode_fxn_scales(**TS_time_loss_scales)
                TS_time_loss = self.TS_time_metrics.ode_fxn(
                    path.TS_time, path
                )
                TS_loss = TS_loss + TS_time_loss 
            if self.TS_region_metrics.ode_fxn is not None:
                self.TS_region_metrics.update_ode_fxn_scales(
                    **TS_region_loss_scales
                )
                TS_region_loss = self.TS_region_metrics.ode_fxn(
                    path.TS_region, path
                )
                TS_loss = TS_loss + TS_region_loss 
            TS_loss.backward()        
        ###########################################

        self.optimizer.step()
        for name, sched in self.ode_fxn_schedulers.items():
            sched.step() 
        for name, sched in self.path_loss_schedulers.items():
            sched.step() 
        for name, sched in self.TS_time_loss_schedulers.items():
            sched.step() 
        for name, sched in self.TS_region_loss_schedulers.items():
            sched.step()
        """
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
        """
        
        ############# Testing ##############
        # Find transition state time
        path.find_TS(path_integral.t, path_integral.y)
        ##############
        self.iteration = self.iteration + 1
        return path_integral
    
    def _TS_max_E(self):
        raise NotImplementedError

    