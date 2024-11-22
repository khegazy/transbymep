import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    SupportsFloat,
    TypedDict,
    Union,
)
from torch import inf, Tensor


class Scheduler:
    def __init__(self, value, last_epoch=-1):
        self.value = value
        self.last_epoch = last_epoch
        self.last_epoch += 1

    def step(self):
        self.last_epoch += 1
    
    def get_value(self):
        if hasattr(self, '_get_closed_form'):
            return self._get_closed_form()
        else:
            return self.value

class Linear(Scheduler):
    def __init__(self, value, start_factor, end_factor, total_iters, last_epoch=-1):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(value, last_epoch)
    
    def _get_closed_form(self):
        return self.value * (self.start_factor + (self.end_factor - self.start_factor) * min(self.last_epoch, self.total_iters) / self.total_iters)
    
class Cosine(Scheduler):
    def __init__(self, value, start_factor, end_factor, total_iters, last_epoch=-1):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(value, last_epoch)
    
    def _get_closed_form(self):
        # return self.eta_min + (self.value - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        return self.value * (self.end_factor + (self.start_factor - self.end_factor) * (1 + math.cos(math.pi * self.last_epoch / self.total_iters)) / 2)
    
# class ReduceOnPlateau(Scheduler):
#     def __init__(self, value, lr_scheduler, factor=0.1, last_epoch=-1):
#         super().__init__(value, last_epoch)
#         self.value = value
#         self.factor = factor

#         self.lr_scheduler = lr_scheduler
#         assert isinstance(self.lr_scheduler, ReduceLROnPlateau)  # TODO: Decouple the loss scheduler from the learning rate scheduler
#         self.lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
    
#     def step(self):
#         print(self.lr, self.lr_scheduler.optimizer.param_groups[0]['lr'])
#         if self.lr != self.lr_scheduler.optimizer.param_groups[0]['lr']:
#             self._reduce_value()
#         self.lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
    
#     def _reduce_value(self):
#         old_value = self.value
#         new_value = old_value * self.factor
#         self.value = new_value

# class IncreaseOnPlateau(Scheduler):
#     def __init__(self, value, lr_scheduler, factor=0.1, last_epoch=-1):
#         super().__init__(value, last_epoch)
#         self.value = 0
#         self.max_value = value
#         self.factor = factor

#         self.lr_scheduler = lr_scheduler
#         assert isinstance(self.lr_scheduler, ReduceLROnPlateau)
#         self.lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
    
#     def step(self):
#         print(self.lr, self.lr_scheduler.optimizer.param_groups[0]['lr'])
#         if self.lr != self.lr_scheduler.optimizer.param_groups[0]['lr']:
#             self._increase_value()
#         self.lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
    
#     def _increase_value(self):
#         old_value = self.value
#         new_value = self.max_value - (self.max_value - old_value) * self.factor
#         self.value = new_value

class ChangeParamOnPlateau(ReduceLROnPlateau):
    def __init__(
        self,
        optimizer,
        integrator,
        num_changes,
        factor_lr=0.1,
        min_lr: Union[List[float], float] = 0,
        max_lr: Union[List[float], float] = inf,
        factor_tol=0.1,
        min_tol: Union[dict[float], float] = 0,
        max_tol: Union[dict[float], float] = inf,
        factor_param=0.1,
        min_param: Union[dict[float], float] = 0,
        max_param: Union[dict[float], float] = inf,
        **kwargs,
    ):
        super().__init__(optimizer, **kwargs)
        self.integrator = integrator

        self.factor_lr = factor_lr
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)
        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} max_lrs, got {len(max_lr)}"
                )
            self.max_lrs = list(max_lr)
        else:
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        self.factor_tol = factor_tol
        if isinstance(min_tol, dict):
            self.min_tols = min_tol
        else:
            self.min_tols = {key: min_tol for key in ['rtol', 'atol']}
        if isinstance(max_tol, dict):
            self.max_tols = max_tol
        else:
            self.max_tols = {key: max_tol for key in ['rtol', 'atol']}
        self.factor_param = factor_param
        if isinstance(min_param, dict):
            self.min_params = min_param
        else:
            self.min_params = {key: min_param for key in integrator.parameters}
        if isinstance(max_param, dict):
            self.max_params = max_param
        else:
            self.max_params = {key: max_param for key in integrator.parameters}

        self.num_changes = num_changes
        self.change_counter = 0
        self.converged = False

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = old_lr * self.factor_lr
            new_lr = max(new_lr, self.min_lrs[i])
            new_lr = min(new_lr, self.max_lrs[i])
            if abs(old_lr - new_lr) > self.eps:
                param_group["lr"] = new_lr
        for key in ['rtol', 'atol']:
            old_tol = self.integrator._integrator.__getattribute__(key)
            new_tol = old_tol * self.factor_tol
            new_tol = max(new_tol, self.min_tols[key])
            new_tol = min(new_tol, self.max_tols[key])
            if abs(old_tol - new_tol) > self.eps:
                self.integrator._integrator.__setattr__(key, new_tol)
        for key in self.integrator.parameters.keys():
            old_param = self.integrator.parameters[key]
            new_param = old_param * self.factor_param
            new_param = max(new_param, self.min_params[key])
            new_param = min(new_param, self.max_params[key])
            if abs(old_param - new_param) > self.eps:
                self.integrator.parameters[key] = new_param
        self.change_counter += 1
        if self.change_counter >= self.num_changes:
            self.converged = True
        self.best = self.mode_worse