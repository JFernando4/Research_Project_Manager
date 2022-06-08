"""
Functions for initialzing network weights and manipulating weights
"""
from collections import namedtuple

import numpy as np
from torch import nn
import torch

distribution = namedtuple("Distribution", ("name", "parameter_values"))


def xavier_init_weights(m, normal=True):
    """
    Initializes weights using xavier initialization
    :param m: torch format for network layer
    :param normal: (bool) whether to use normal initialization, uses uniform is false
    :return: None, operation is done in-place
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        if normal:
            nn.init.xavier_normal_(m.weight)
        else:
            nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def init_weights_kaiming(m, nonlinearity='relu', normal=True):
    """
    Initializes weights using kaiming initialization
    :param m: torch format for network layer
    :param nonlinearity: (str) specifies the type of activation used in layer m
    :param normal: (bool) whether to use normal initialization, uses uniform is false
    :return: None, operation is done in-place
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        if normal:
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def init_weights_normal(m, parameter_values: tuple):
    """
    Initializes weights using normal distribution
    :param m: torch format for network layer
    :param parameter_values: (tuple) containing the mean and standard deviation of the normal distribution
    :return: None, operation is done in-place
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=parameter_values[0], std=parameter_values[1])
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def init_weights_uniform(m, parameter_values: tuple):
    """
    Initializes weights using uniform distribution
    :param m: torch format for network layer
    :param parameter_values: (tuple) containing the lower and upper bound of the uniform distributinon
    :return: None, operation is done in-place
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        lower_bound, upper_bound = parameter_values
        nn.init.uniform_(m.weight, a=lower_bound, b=upper_bound)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def get_initialization_function(dist: distribution):
    """
    returns one of the many initialization functions according to the given distribution
    :param dist: (named tuple) specifies the name and parameter_values of the distribution. The options are:
                            ## name ##              ## parameter_values ##
                            "xavier"                (bool, None) — specifies whether to use a normal distribution
                            "kaiming"               (bool, str) — (whether to use normal distribution,
                                                                   name of the activation function)
                            "normal"                (float, float) — mean and standard deviation
                            "uniform"               (float, float) — upper and lower bounds
    :return: lambda function corresponding to the initialization method
    """
    if dist.name == "xavier":
        return lambda z: xavier_init_weights(z, normal=dist.parameter_values[0])
    elif dist.name == "kaiming":
        return lambda z: init_weights_kaiming(z, normal=dist.parameter_values[0], nonlinearity=dist.parameter_values[1])
    elif dist.name == "normal":
        return lambda z: init_weights_normal(z, parameter_values=dist.parameter_values)
    elif dist.name == "uniform":
        return lambda z: init_weights_uniform(z, parameter_values=dist.parameter_values)
    else:
        raise ValueError("{0} is not a valid distribution!".format(dist.name))


def scale_weights(m, scale=0.9):
    """
    Multiplies weights by a factor
    :param m: torch format for network layer
    :param scale: (float) scaling factor
    :return: None, operation is done in-place
    """
    with torch.no_grad():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            m.weight.data *= scale


def add_noise_to_weights(m, stddev=0.01):
    """
    Adds noise to the weights of the network
    :param m: torch format for network layer
    :param stddev: (float) standard deviation of the noise
    :return: None, operation is done in-place
    """
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size(), device=m.weight.device) * stddev)


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
            return lambda z: torch.normal(mean=torch.tensor((mean, ) * z).to(dtype=torch.float32),
                                          std=torch.tensor((std, ) * z).to(dtype=torch.float32))
        else:
            return lambda z: np.random.normal(mean, std, z)
    elif name == "uniform":
        """ parameter value format: (lower bound, upper bound) """
        lower_bound = parameter_values[0]
        upper_bound = parameter_values[1]
        if use_torch:
            return lambda z: torch.rand(z, dtype=torch.float32) * (upper_bound - lower_bound) + lower_bound
        else:
            return lambda z: np.random.uniform(lower_bound, upper_bound, z)
    else:
        raise ValueError("{0} is not a valid distribution!".format(name))
