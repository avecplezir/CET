import torch
import numpy as np
import math

def sparse_init(tensor, sparsity, type='uniform'):

    if tensor.ndimension() == 2:
        fan_out, fan_in = tensor.shape

        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if type == 'uniform':
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif type == 'normal':
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
            for col_idx in range(fan_out):
                row_indices = torch.randperm(fan_in)
                zero_indices = row_indices[:num_zeros]
                tensor[col_idx, zero_indices] = 0
        return tensor

    elif tensor.ndimension() == 4:
        channels_out, channels_in, h, w = tensor.shape
        fan_in, fan_out = channels_in*h*w, channels_out*h*w

        num_zeros = int(math.ceil(sparsity * fan_in))

        with torch.no_grad():
            if type == 'uniform':
                tensor.uniform_(-math.sqrt(1.0 / fan_in), math.sqrt(1.0 / fan_in))
            elif type == 'normal':
                tensor.normal_(0, math.sqrt(1.0 / fan_in))
            else:
                raise ValueError("Unknown initialization type")
            for out_channel_idx in range(channels_out):
                indices = torch.randperm(fan_in)
                zero_indices = indices[:num_zeros]
                tensor[out_channel_idx].reshape(channels_in*h*w)[zero_indices] = 0
        return tensor
    else:
        raise ValueError("Only tensors with 2 or 4 dimensions are supported")


def transform_obs(x, env_id):
    if 'MinAtar' in env_id:
        x = x.permute(0, 3, 1, 2).float()
    elif 'MiniGrid' in env_id:
        x = x.permute(0, 3, 1, 2).float() / 255.0
    elif 'Atari' in env_id:
        x = x / 255.0

    return x


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    # torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.kaiming_normal_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
