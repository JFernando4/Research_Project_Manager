"""
Functions for initialzing network weights and manipulating weights
"""
from torch import nn
import torch


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
        m.bias.data.fill_(0.0)


def init_weights_kaiming(m, nonlinearity='relu', normal=True):
    """
    Initializes weights using kaiming initialization
    :param m: torch format for network layer
    :param nonlinearity: (str) specifies the type of activation used in layer m
    :param normal: (bool) whether to use normal initialization, uses uniform is false
    :return: None, operation is done in-place
    """
    if m is isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        if normal:
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
        m.bias.data.fill_(0.0)


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
