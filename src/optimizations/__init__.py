from .gradient_descent import gradientDescent

optimizer_dict = {
    "gradient_descent" : gradientDescent
}

def get_optimizer(name):
    assert(name.lower() in optimizer_dict)
    return optimizer_dict[name.lower()]