import math
import torch
import torch.nn as nn
from torch import empty
from torch.nn import Parameter
from torch.nn.functional import relu
from torch.nn import ReLU, Softmax

def ff_layer(in_features: int, out_features: int, activation_function: relu, device: str="cuda"):
    weight = Parameter(empty((out_features, in_features), device=device))
    bias = Parameter(empty(out_features, device=device))

    def weight_and_bias_initialization():
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)
    weight_and_bias_initialization()

    def linear_computation(x: torch.Tensor):
        return torch.matmul(x, weight.t()) + bias

    def forward_pass(x: torch.Tensor) -> torch.Tensor:
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return activation_function(linear_computation(x))
    
    return forward_pass, weight, bias

def linear_layer(in_features: int, out_features: int, device: str, activation_function: str):
    if activation_function == 'relu':
        act_func = ReLU()
    elif activation_function == 'softmax':
        act_func = Softmax(dim=-1)
    
    weight = Parameter(empty((out_features, in_features), device=device))
    bias = Parameter(empty(out_features, device=device))

    def weight_and_bias_initialization():
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(bias, -bound, bound)
    weight_and_bias_initialization()

    def linear_computation(x: torch.Tensor):
        return act_func(torch.matmul(x, weight.t()) + bias)
    
    return linear_computation, weight, bias