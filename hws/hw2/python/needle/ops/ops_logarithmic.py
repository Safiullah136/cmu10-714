from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=1, keepdims=True)
        sum_exp = array_api.sum(array_api.exp(Z - max_z), axis=1, keepdims=True)
        log_sum_exp = array_api.log(sum_exp) + max_z
        return Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        axes = (1, )
        z = node.inputs[0]
        max_z = Tensor(z.realize_cached_data().max(axis=axes, keepdims=True))
        exp_z = exp(z - max_z)
        sum_exp_z = Tensor(array_api.sum(exp_z.realize_cached_data(), axes, keepdims=True))
        sum_out_grad = Tensor(out_grad.realize_cached_data().sum(axes, keepdims=True))
        return out_grad - sum_out_grad * (exp_z / sum_exp_z)
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=self.axes, keepdims=True)
        out = array_api.sum(array_api.exp(Z - max_z), axis=self.axes)
        out = array_api.log(out) + array_api.squeeze(max_z)
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = Tensor(z.realize_cached_data().max(axis=self.axes, keepdims=True))
        exp_z = exp(z - max_z)
        sum_exp_z = exp_z.sum(self.axes)
        grad = out_grad / sum_exp_z

        new_shape = array_api.array(z.shape)
        if self.axes is None:
          new_shape[:] = 1
        else:  
          new_shape[list(self.axes)] = 1
        
        return grad.reshape(new_shape) * exp_z
        
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)