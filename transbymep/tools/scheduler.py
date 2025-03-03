import math
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class SchedulerBase:
    def __init__(self, value=1.0, last_epoch=-1):
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

class Linear(SchedulerBase):
    def __init__(self, start_factor, end_factor, total_iters, **kwargs):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(**kwargs)
    
    def _get_closed_form(self):
        return self.value * (self.start_factor + (self.end_factor - self.start_factor) * min(self.last_epoch, self.total_iters) / self.total_iters)
    
class Cosine(SchedulerBase):
    def __init__(self, start_factor, end_factor, total_iters, **kwargs):
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        super().__init__(**kwargs)
    
    def _get_closed_form(self):
        # return self.eta_min + (self.value - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        return self.value * (self.end_factor + (self.start_factor - self.end_factor) * (1 + math.cos(math.pi * self.last_epoch / self.total_iters)) / 2)
    
# class ReduceOnPlateau(SchedulerBase):
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

# class IncreaseOnPlateau(SchedulerBase):
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


SCHEDULER_DICT = {
    'linear' : Linear,
    'cosine' : Cosine,
    # 'reduce_on_plateau' : ReduceOnPlateau,
    # 'increase_on_plateau' : IncreaseOnPlateau 
}

LR_SCHEDULER_DICT = {
    "step" : lr_scheduler.StepLR,
    "linear" : lr_scheduler.LinearLR,
    "multi_step" : lr_scheduler.MultiStepLR,
    "exponential" : lr_scheduler.ExponentialLR,
    "cosine" : lr_scheduler.CosineAnnealingLR,
    "cosine_restart" : lr_scheduler.CosineAnnealingWarmRestarts,
    "one_cycle" : lr_scheduler.OneCycleLR,
    "reduce_on_plateau" : lr_scheduler.ReduceLROnPlateau,
}

def get_schedulers(scheduler_params):
    schedulers = {}
    if scheduler_params is None:
        return schedulers
    
    for name, param_dict in scheduler_params.items():
        assert 'name' in param_dict, f"Must specify name of scheduler: {list(SCHEDULER_DICT.keys())}"
        assert param_dict['name'].lower() in SCHEDULER_DICT,\
            f"Cannot find scheduler {param_dict['name']}, options are {list(SCHEDULER_DICT.keys())}"
        sched_name = param_dict.pop('name').lower()
        schedulers[name] = SCHEDULER_DICT[sched_name](**param_dict)
    return schedulers

def get_lr_scheduler(optimizer, param_dict):
    assert 'name' in param_dict, f"Must specify name of scheduler: {list(LR_SCHEDULER_DICT.keys())}"
    assert param_dict['name'].lower() in LR_SCHEDULER_DICT,\
        f"Cannot find scheduler {param_dict['name']}, options are {list(LR_SCHEDULER_DICT.keys())}"
    sched_name = param_dict.pop('name').lower()
    return LR_SCHEDULER_DICT[sched_name](optimizer, **param_dict)