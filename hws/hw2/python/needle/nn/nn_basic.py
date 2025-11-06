"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, dtype=dtype, device=device, requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, dtype=dtype, device=device, requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X.matmul(self.weight)
        if self.bias is not None:
          out += self.bias.broadcast_to(out.shape)
        return out 
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return X.reshape( (X.shape[0], -1) )
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module.forward(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(logits.shape[1], y)
        log_sum_exp = ops.logsumexp(logits, (1, ))
        return 1 / y.shape[0] * (log_sum_exp.sum() - (logits * one_hot).sum())
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device)
        self.running_var = init.ones(dim, device=device)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          mean_x = ( x.sum( (0, ) ) / x.shape[0] )
          b_mean_x = mean_x.broadcast_to( x.shape )
          x_minus_mean_x = x - b_mean_x
          var_x = ( (x_minus_mean_x ** 2).sum((0, )) / x.shape[0] )
          b_var_x = var_x.broadcast_to( x.shape )

          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x

          d = (b_var_x + self.eps) ** 0.5 
          return self.weight.broadcast_to(x.shape) * ( x_minus_mean_x / d ) + self.bias.broadcast_to(x.shape)

        else:
          n = x - self.running_mean.broadcast_to( x.shape )
          d = (self.running_var.broadcast_to( x.shape ) + self.eps) ** 0.5 
          return self.weight.broadcast_to(x.shape) * ( n / d ) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean_x = ( x.sum( (1, ) ) / x.shape[1] ).reshape( (x.shape[0], 1) ).broadcast_to( x.shape )
        x_minus_mean_x = x - mean_x
        var_x = ( (x_minus_mean_x ** 2).sum((1, )) / x.shape[1] ).reshape( (x.shape[0], 1) ).broadcast_to( x.shape )

        d = (var_x + self.eps) ** 0.5 
        return self.weight.broadcast_to(x.shape) * ( x_minus_mean_x / d ) + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
          return x

        t = init.randb(*x.shape, p=1-self.p)
        return (x * t) / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
