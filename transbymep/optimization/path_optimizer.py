import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch_optimizer import Adahessian
from transbymep.tools import scheduler

optimizer_dict = {
    "sgd" : optim.SGD,
    "adagrad" : optim.Adagrad,
    "adam" : optim.Adam,
    "lbfgs" : optim.LBFGS,
    "adahessian" : Adahessian,
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
    # "reduce_on_plateau" : scheduler.ReduceOnPlateau,
    # "increase_on_plateau" : scheduler.IncreaseOnPlateau,
    "plateau" : scheduler.ChangeParamOnPlateau,
}

class PathOptimizer():
    def __init__(
            self,
            name,
            path,
            loss_name,
            device='cpu',
            **config
        ):
        self.loss_name = loss_name
        self.device=device
        
        # Initialize optimizer
        name = name.lower()
        # self.optimizer = optimizer_dict[name](path.parameters(), **config)
        trainable_params = filter(lambda p: p.requires_grad, path.parameters())
        self.optimizer = optimizer_dict[name](trainable_params, **config)
        self.scheduler = None
        self.loss_scheduler = None
        self.converged = False

    def set_scheduler(self, name, **config):
        name = name.lower()
        if name not in scheduler_dict:
            raise ValueError(f"Cannot handle scheduler type {name}, either add it to scheduler_dict or use {list(scheduler_dict.keys())}")
        self.scheduler = scheduler_dict[name](self.optimizer, **config)

    def set_loss_scheduler(self, **kwargs):
        # self.loss_scheduler = {}
        # for key, value in kwargs.items():
        #     name = value.pop('name').lower()
        #     if name not in loss_scheduler_dict:
        #         raise ValueError(f"Cannot handle loss scheduler type {name}, either add it to loss_scheduler_dict or use {list(loss_scheduler_dict.keys())}")
        #     if name == "reduce_on_plateau" or name == "increase_on_plateau":
        #         self.loss_scheduler[key] = loss_scheduler_dict[name](lr_scheduler=self.scheduler, **value)
        #     else:
        #         self.loss_scheduler[key] = loss_scheduler_dict[name](**value)
        name = kwargs.pop('name').lower()
        if name not in loss_scheduler_dict:
            raise ValueError(f"Cannot handle loss scheduler type {name}, either add it to loss_scheduler_dict or use {list(loss_scheduler_dict.keys())}")
        self.loss_scheduler = loss_scheduler_dict[name](**kwargs)
    
    def optimization_step(
            self,
            path,
            integrator,
            t_init=torch.tensor([0.], dtype=torch.float64),
            t_final=torch.tensor([1.], dtype=torch.float64)
        ):
        t_init = t_init.to(torch.float64).to(self.device)
        t_final = t_final.to(torch.float64).to(self.device)
        if isinstance(self.optimizer, optim.LBFGS):
            def closure():
                self.optimizer.zero_grad()
                path_integral = integrator.path_integral(
                    path, self.loss_name, t_init=t_init, t_final=t_final
                )
                loss = path_integral.integral

                if (integrator.parameters is not None) and ('force_scale' in integrator.parameters):
                    time = path_integral.t.flatten()
                    time = time[len(time)//10:-len(time)//10]
                    path_output = path(time, return_force=True)
                    # path_output = path(time, return_energy=True, return_force=True)
                    force = torch.linalg.norm(path_output.path_force, dim=-1).min()
                    # force = path_output.path_force[path_output.path_energy.argmax()]
                    # force = torch.linalg.norm(force)
                    loss = loss + force * integrator.parameters['force_scale']

                loss.backward()

                # grad_norm = clip_grad_norm_(path.parameters(), 20)
                # print("grad_norm", grad_norm)

                return loss
            self.optimizer.param_groups[0]["line_search_fn"] = "strong_wolfe" if torch.randint(0, 2, (1,)).item() == 0 else None
            loss = self.optimizer.step(closure)
            path_integral = integrator.integral_output

            # flat_grad = self.optimizer._gather_flat_grad()
            # print(flat_grad.abs().max())
        elif isinstance(self.optimizer, Adahessian):
            self.optimizer.zero_grad()
            path_integral = integrator.path_integral(
                path, self.loss_name, t_init=t_init, t_final=t_final
            )
            loss = path_integral.integral
            loss.backward(create_graph=True)
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
            path_integral = integrator.path_integral(
                path, self.loss_name, t_init=t_init, t_final=t_final
            )
            #for n, prm in path.named_parameters():
            #    print(n, prm.grad)
            #print("path integral", path_integral)
            if integrator._integrator.max_batch is None:
                #TODO: Better to change this to backprop if not detached
                loss = path_integral.integral

                # time = path_integral.t.flatten()
                # time = time[len(time)//10:-len(time)//10]
                # path_output = path(time, return_force=True)
                # # path_output = path(time, return_energy=True, return_force=True)
                # force = torch.linalg.norm(path_output.path_force, dim=-1).min()
                # # force = path_output.path_force[path_output.path_energy.argmax()]
                # # force = torch.linalg.norm(force)
                # loss = loss + force * integrator.parameters['force_scale']

                loss.backward()
                # (path_integral.integral**2).backward()
            self.optimizer.step()
        # if self.scheduler is not None:
        #     if isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
        #         self.scheduler.step(loss)
        #         # time = path_integral.t.flatten()
        #         # time = time[len(time)//10:-len(time)//10]
        #         # force = path(time, return_force=True).path_force
        #         # self.scheduler.step(torch.linalg.norm(force, dim=-1).min())
        #         if self.optimizer.param_groups[0]['lr'] <= self.scheduler.min_lrs[0]:
        #             self.converged = True
        #     else:
        #         self.scheduler.step()
        #     print("Learning rate:", self.optimizer.param_groups[0]['lr'])
        if self.loss_scheduler is not None:
            self.loss_scheduler.step(loss)
            # metric_parameters = {key: loss_scheduler.get_value() for key, loss_scheduler in self.loss_scheduler.items()}
            # integrator.update_metric_parameters(metric_parameters)
            if self.loss_scheduler.converged:
                self.converged = True
            print("Loss:", loss.item(), "Best:", self.loss_scheduler.best, "Bad epochs:", self.loss_scheduler.num_bad_epochs)
            print("Learning rate:", self.optimizer.param_groups[0]['lr'])
            print("Tolerance:", integrator._integrator.rtol, integrator._integrator.atol)
            print("Loss parameter:", integrator.parameters)

        return path_integral

    