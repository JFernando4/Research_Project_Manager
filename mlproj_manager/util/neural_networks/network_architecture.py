"""
Functions related to building the network architecture and optimizing the network weights
"""
import numpy as np
import torch
from torch import optim
from torch import relu, sigmoid, tanh
from collections import namedtuple


layer = namedtuple("Layer", ("type", "parameters", "gate"))


def get_conv_layer_output_dims(h_in, w_in, kernel_size, stride, padding=(0, 0), dilatation=(1, 1)):
    h_out = np.floor(((h_in - 1) + (2 * padding[0]) - (dilatation[0] * (kernel_size[0] - 1))) / stride[0]) + 1
    w_out = np.floor(((w_in - 1) + (2 * padding[1]) - (dilatation[1] * (kernel_size[1] - 1))) / stride[1]) + 1
    return int(h_out), int(w_out)


def get_optimizer(optimizer: str, nn_parameters, **kwargs):
    """
    :return: pytorch optimizer object
    """
    stepsize = 0.001 if "stepsize" not in kwargs.keys() else kwargs['stepsize']
    weight_decay = 0.0 if "weight_decay" not in kwargs.keys() else kwargs["weight_decay"]
    if optimizer == "adam":
        beta1 = 0.9 if "beta1" not in kwargs.keys() else kwargs['beta1']
        beta2 = 0.99 if "beta2" not in kwargs.keys() else kwargs['beta2']
        return optim.Adam(nn_parameters, lr=stepsize, betas=(beta1, beta2), weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(nn_parameters, lr=stepsize, weight_decay=weight_decay)
    else:
        raise ValueError("{0} is not a valid optimizer!".format(optimizer))


def get_activation(name):
    if name == "relu":
        return relu
    elif name == "tanh":
        return tanh
    elif name == "sigmoid":
        return sigmoid
    elif name is None:
        return lambda x: x
    else:
        raise ValueError("{0} is not a valid activation!")


def get_activation_module(name: str):
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "tanh":
        return torch.nn.Tanh()
    elif name == "sigmoid":
        return torch.nn.Sigmoid()
    elif name == "leaky_relu":
        return torch.nn.LeakyReLU()
    else:
        raise ValueError("{0} is not a valid activation!".format(name))
