import os
import yaml

# from .wolfe_schlegel import WolfeSchlegel
# from .muller_brown import MullerBrown
# from .schwefel import Schwefel
# from .constant import Constant
# from .newtonnet import NewtonNetPotential
# from .escaip import EScAIPPotential


# potential_dict = {
#     "wolfe_schlegel": WolfeSchlegel,
#     "muller_brown": MullerBrown,
#     "schwefel": Schwefel,
#     "constant": Constant,
#     "newtonnet": NewtonNetPotential,
#     "escaip": EScAIPPotential,
# }

def potential_dict(name):
    if name == "wolfe_schlegel":
        from .wolfe_schlegel import WolfeSchlegel
        return WolfeSchlegel
    elif name == "muller_brown":
        from .muller_brown import MullerBrown
        return MullerBrown
    elif name == "schwefel":
        from .schwefel import Schwefel
        return Schwefel
    elif name == "constant":
        from .constant import Constant
        return Constant
    elif name == "newtonnet":
        from .newtonnet import NewtonNetPotential
        return NewtonNetPotential
    elif name == "escaip":
        from .escaip import EScAIPPotential
        return EScAIPPotential
    else:
        raise ValueError(f"Cannot handle potential type {name}, either add it to potential_dict or use {list(potential_dict.keys())}")


# def import_potential_config(
#         name,
#         tag="",
#         dir=os.path.join(
#             os.path.dirname(os.path.abspath(__file__)),
#             "configs"
#         ),
#         is_expected=False
#     ):
#     filename = name
#     filename += f"_{tag}.yaml" if tag != "" else ".yaml"
#     address = os.path.join(dir, filename)
#     if os.path.exists(address):
#         with open(address, 'r') as file:
#             loaded_yaml = yaml.safe_load(file)
#         return loaded_yaml
#     elif is_expected:
#         raise ImportError(f"Cannot find required file {address}")
#     else:
#         ImportWarning(f"Cannot find file {address}, running without it")
#         return {}


def get_potential(
        potential,
        # tag="",
        # config_dir=os.path.join(
        #         os.path.dirname(os.path.abspath(__file__)),
        #         "configs"
        #     ),
        # expect_config=False,
        **kwargs
    ):
    # assert potential.lower() in potential_dict, f"Potential {potential} not found"
    # config_filename = potential
    # config_filename += f"_{tag}.yaml" if tag != "" else ".yaml"
    # print(potential)
    # config = import_potential_config(
    #     potential, tag, dir=config_dir, is_expected=expect_config
    # )
    # return potential_dict[potential](**config, **kwargs, config_dir=config_dir)
    # return potential_dict[potential](**kwargs)
    return potential_dict(potential)(**kwargs)
