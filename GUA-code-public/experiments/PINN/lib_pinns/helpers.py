import torch,random,os
import numpy as np

def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

# Adapted from a reference PINN implementation of the Schrodinger equation.
def derivative(y: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    for i in range(order):
        y = torch.autograd.grad(
            y, x, grad_outputs = torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]
    return y
    
def package_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+os.sep
    
