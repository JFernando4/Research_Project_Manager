# from third party packages:
import torch
import torch.nn as nn
import numpy as np
# from project files:
from mlproj_manager.util.neural_networks.network_architecture import layer, get_conv_layer_output_dims, get_activation_module


class GenericDeepNet(nn.Module):
    """ Builds a conv net with a specified number of layers and dimensions """

    def __init__(self, architecture, image_dims, use_bias=True):
        """
        :param architecture: a list of named tuples where each tuple represents a layer. Each named tuple must contain
                             the following names:
            - type: string specifying the type of layer. Available options:
                        { "conv2d", "maxpool", "linear", "flatten"}
            - parameters: a tuple containing the parameters of the layer. The parameters for each layer type should be:
                           - conv2d: in_channels (int), out_channels (int), kernel_size (int,) x 2, stride (int,) x 2
                           - maxpool: kernel_size (int, ) x 2, stride (int, ) x 2
                           - linear: None (in_features are computed based on the previous layer), out_features (int)
                           - flatten: empty tuple
            - gate: a string specifying the gate function for the layer. Available options:
                      { "relu", "sigmoid", "tanh", None } (None corresponds to the identity function f(x) = x)
        :param image_dims:  the width and height of the input images (int, ) x 2
        :param use_bias: (bool) whether to use a bias term in  each layer
        """
        super(GenericDeepNet, self).__init__()

        # check that each layer has the correct format
        assert all(isinstance(a_layer, layer) for a_layer in architecture)

        self.architecture = architecture
        self.image_dims = image_dims
        self.use_bias = use_bias

        # create list of modules
        self.network_module_list = nn.ModuleList()
        self.create_modules()

    def create_modules(self):
        out_channels = 1
        h_out, w_out = self.image_dims
        for a_layer in self.architecture:
            if a_layer.type == "conv2d":
                in_channels, out_channels, kernel_size, stride = a_layer.parameters
                self.network_module_list.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                                          bias=self.use_bias))
                h_out, w_out = get_conv_layer_output_dims(h_out, w_out, kernel_size, stride)
            elif a_layer.type == "maxpool":
                kernel_size, stride = a_layer.parameters
                self.network_module_list.append(nn.MaxPool2d(kernel_size, stride))
                h_out, w_out = get_conv_layer_output_dims(h_out, w_out, kernel_size, stride)
            elif a_layer.type == "linear":
                _, out_features = a_layer.parameters
                in_features = h_out * w_out * out_channels
                self.network_module_list.append(nn.Linear(in_features, out_features, bias=self.use_bias))
                h_out, w_out, out_channels = (out_features, 1, 1)
            elif a_layer.type == "flatten":
                assert len(a_layer.parameters) == 0
                self.network_module_list.append(nn.Flatten())
            elif a_layer.type == "dropout":
                self.network_module_list.append(torch.nn.Dropout(p=a_layer.parameters))
            else:
                raise ValueError("{0} is not a valid layer!".format(a_layer))

            if a_layer.gate is not None:
                self.network_module_list.append(get_activation_module(a_layer.gate))

    def forward(self, obs, return_activations=False):
        x = obs
        activations = ()

        for i, module in enumerate(self.network_module_list):
            # apply layer
            x = module(x)
            # apply mask: this is for shrinking or masking out certain features
            is_activation = isinstance(module, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh))
            if return_activations and is_activation:
                activations += (x, )

        if return_activations:
            return x, activations
        return x

    def get_module_list(self):
        return self.network_module_list


def main():
    """ Test: training the network """
    # import extra modules
    from mlproj_manager.problems.supervised_learning import CifarDataSet
    from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, RandomGaussianNoise, RandomErasing
    from mlproj_manager.definitions import ROOT
    import os
    import torch.optim as optim
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from mlproj_manager.util.neural_networks.weights_initialization_and_manipulation import xavier_init_weights

    # training parameters
    batch_size = 4
    num_epochs = 100
    stepsize = 0.01
    checkpoint = 25
    current_batch = 0
    num_classes = 5

    torch.random.manual_seed(0)
    """ Test: setting up the network """
    architecture = [
        layer(type="conv2d",    parameters=(3, 64, (3,3), (1,1)),   gate="relu"),       # conv 1
        layer(type="maxpool",   parameters=((2,2), (2,2)),          gate=None),         # max pool 1
        layer(type="conv2d",    parameters=(64, 32, (3,3), (1,1)),  gate="relu"),       # conv 2
        layer(type="maxpool",   parameters=((2,2), (1,1)),          gate=None),         # max pool 2
        layer(type="flatten",   parameters=(),                      gate=None),         # flatten
        layer(type="linear",    parameters=(None, 1024),             gate='relu'),       # feed forward 1
        layer(type="linear",    parameters=(None, num_classes),              gate=None)          # output layer
    ]

    network = GenericDeepNet(architecture, (32, 32))
    network.apply(lambda z: xavier_init_weights(z, normal=True))

    image = torch.normal(0, 0.5, (3, 32, 32))
    output = network.forward(image[None, :], return_activations=False)
    print("Output: \n\t", output, "\nSum: ", torch.sum(output))


if __name__ == "__main__":
    main()
