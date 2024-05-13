import os
import hostlist
import torch
import torch.distributed as dist

###########################################################################################
# Slurm environment setup for distributed training.
# This code is refactored from rsarm's contribution at:
# https://github.com/Lumi-supercomputer/lumi-reframe-tests/blob/main/checks/apps/deeplearning/pytorch/src/pt_distr_env.py
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

class DistributedEnvironment():
    is_distributed : bool
    is_slurm : bool
    is_master : bool
    rank : int
    local_rank : int
    world_size : int
    seed_offset : int
    device_type : str
    device : str

    def __init__(
        self,
        backend : str = None,
        device_type : str = 'cuda',
        master_addr : str = None,
        master_port : str = None,
        is_slurm : bool = False
    ): 
        self.is_slurm = is_slurm
        self.device_type = device_type
        self.backend = backend
        if backend is None:
            if device_type.lower() == 'cpu':
                self.backend = 'gloo'
            elif device_type.lower() == 'cuda':
                self.backend = 'nccl'
            else:
                raise ValueError("Can't assign a default backend to device_type {device_type}, use 'cpu' or 'cuda'.")
        self.backend = self.backend.lower()

        # Determing if process is distributed
        self.check_distributed()

        # Initialize process
        if self.is_distributed:
            self.init_distributed_process(master_addr, master_port)
        else:
            self.init_single_process()
        
        # Set device
        self.set_device()


    def check_distributed(self):
        if self.is_slurm:
            self.is_distributed = int(os.environ['SLURM_NTASKS']) > 1
        else:
            self.is_distributed = int(os.environ.get('RANK', -1)) != -1
        
        # Check if torch.distributed is available
        if self.is_distributed:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available") 
            if self.backend == 'nccl':
                assert dist.is_nccl_available()
            elif self.backend == 'mpi':
                assert dist.is_mpi_available()
            elif self.backend == 'gloo':
                assert dist.is_gloo_available()
            else:
                raise SyntaxWarning(f"Cannot check if {self.backend} is available.")
        
        return self.is_distributed
 
    
    def init_single_process(self):
        self.is_master = True
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.seed_offset = 0


    def init_slurm_environment(self):
        hostname = hostlist.expand_hostlist(os.environ['SLURM_JOB_NODELIST'])[0]
        os.environ['MASTER_ADDR'] = hostname
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '33333')
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        return os.environ['MASTER_ADDR'], os.environ['MASTER_PORT']

   
    def init_distributed_process(self, master_addr, master_port):
        if self.is_slurm:
            addr, port = self.init_slurm_environment()
            master_addr = addr if master_addr is None else master_addr
            master_port = port if master_port is None else master_port
        
        # Creating process configuration file
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.is_master = self.rank == 0
        self.seed_offset = self.rank

        # Set master IP address and port
        master_addr_port_none = (master_addr is None)\
            and (master_port is None)
        master_addr_port_not_none = (master_addr is not None)\
            and (master_port is not None)
        if not master_addr_port_none and not master_addr_port_not_none:
            raise ValueError("Master address and port must both be specified")
        
        if master_addr_port_not_none:
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = master_port
        self.master_addr = os.environ.get('MASTER_ADDR', None)
        self.master_port = os.environ.get('MASTER_PORT', None)
        
        # Initialize distributed process
        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            init_method='env://',
        )


    def set_device(self):
        print(f"Process {self.rank}: Number of visible CPUs: {os.cpu_count()}")
        if self.device_type == "cuda":
            print(f"Process {self.rank}: Number of visible GPUs: {torch.cuda.device_count()}")
        # If node has multiple GPUs visible by process then select GPU by local
        # rank. Otherwise, process will only see a single GPU so use 'cuda:0'
        is_cuda_with_GPUs = self.device_type.lower() == "cuda"
        print(is_cuda_with_GPUs, torch.cuda.device_count())
        is_cuda_with_GPUs = is_cuda_with_GPUs and torch.cuda.device_count() > 1
        is_cpu_with_CPUs = self.device_type.lower() == "cpu"
        is_cpu_with_CPUs = is_cpu_with_CPUs and os.cpu_count() > 1
        if is_cuda_with_GPUs or is_cpu_with_CPUs:
            self.device = f"{self.device_type}:{self.local_rank}"
            self.device_ids = [self.local_rank]
        else:
            self.device = f"{self.device_type}:0"
            self.device_ids = [0]
        print(f"Process {self.rank}: Running process on {self.device}")
 
        # Initialize cuda
        if self.device_type == "cuda":
            torch.cuda.init()
        
        # Set device
        if self.device_type == "cuda":
            torch.cuda.set_device(self.device)
        elif self.device_type == "cpu":
            print("DEBUG skipping for now")
            #torch.set_default_device(self.device)
        else:
            raise ValueError(f"Cannot handle device type {self.device_type}")


    def end_process(self):
        if self.is_distributed:
            dist.destroy_process_group()
    

    def get_cuda_memory_usage(self):
        if self.device_type != "cuda":
            return -1
        return torch.cuda.mem_get_info()
        return torch.cuda.get_device_properties(self.device).total_memory
        