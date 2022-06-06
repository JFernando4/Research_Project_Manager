"""
Functions related to building the network architecture and optimizing the network weights
"""
import numpy as np
import torch
from torch import optim
from torch.nn.functional import relu, tanh, sigmoid
from collections import namedtuple


layer = namedtuple("Layer", ("type", "parameters", "gate"))
distribution = namedtuple("Distribution", ("name", "parameter_values"))


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
        return torch.jit.script(relu)
    elif name == "tanh":
        return torch.jit.script(tanh)
    elif name == "sigmoid":
        return torch.jit.script(sigmoid)
    elif name is None:
        return lambda x: x
    else:
        raise ValueError("{0} is not a valid activation!")


def get_distribution_function(name: str, parameter_values: tuple, use_torch=False):
    """
    Returns a distribution function according to the specified parameters
    :param name: (str) name of the distribution
    :param parameter_values: (tuple) parameter values used for generating a distribution
    :param use_torch: (bool) whether to generate a distribution using torch
    :return: lambda function that takes an input integer and returns an array with that many random numbers
             generated using the specified distribution
    """
    if name == "normal":
        """ parameter value format: (mean, standard deviation) """
        mean = parameter_values[0]
        std = parameter_values[1]
        if use_torch:
            return lambda z: torch.normal(mean=torch.tensor((mean, ) * z).double(),
                                          std=torch.tensor((std, ) * z).double())
        else:
            return lambda z: np.random.normal(mean, std, z)
    elif name == "uniform":
        """ parameter value format: (lower bound, upper bound) """
        lower_bound = parameter_values[0]
        upper_bound = parameter_values[1]
        if use_torch:
            return lambda z: torch.rand(z) * (upper_bound - lower_bound) + lower_bound
        else:
            return lambda z: np.random.uniform(lower_bound, upper_bound, z)
    else:
        raise ValueError("{0} is not a valid distribution!".format(name))