from typing import Optional
import torch
from torch import Tensor

class StableMax(torch.nn.Module):
    
    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        def stablemax(x, dim):
            y = torch.where(x>0, x+1, 1/(1-x))
            return y/y.sum(dim, keepdim=True)
        
        self.stablemax = stablemax

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "dim"):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return self.stablemax(input, self.dim) #, _stacklevel=5)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"