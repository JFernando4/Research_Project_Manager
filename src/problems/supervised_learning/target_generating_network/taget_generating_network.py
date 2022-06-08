"""
Target generating network similar to the bit-flipping network from Rich.
A network is generated randomly with a pre-specified number of hidden layers and neurons and activation function.
Targets are generated by sampling random inputs according to a specified input distribution.
"""
import numpy as np
import torch
# from project files:
from src.util.neural_networks import get_activation
from src.util import get_distribution_function, distribution


class FeedForwardTargetGeneratingNetwork:
    
    def __init__(self,
                 input_dims: int,
                 num_outputs: int,
                 num_hidden_layers: int,
                 num_hidden_units: int,
                 gate_function: str,
                 input_distribution: distribution,
                 weight_distribution: distribution,
                 use_torch=False):
        """
        :param input_dims: (int) input dimensions
        :param num_outputs: (int) number of output units
        :param num_hidden_layers: (int) number of hidden layers
        :param num_hidden_units: (int) number of hidden units
        :param gate_function: (str) activation function for hidden units
        :param input_distribution: (distribution) input distribution
        :param weight_distribution: (distribution) weight distribution
        :param use_torch: (bool) indicates whether to use torch or not
        """
        self.input_dims = input_dims
        self.num_hidden_layers = num_hidden_layers
        self.activation_functions_str = gate_function
        self.activation_functions = get_activation(gate_function)
        self.input_dist = get_distribution_function(input_distribution.name, input_distribution.parameter_values,
                                                    use_torch)

        weight_diist = get_distribution_function(weight_distribution.name, weight_distribution.parameter_values,
                                                 use_torch)
        self.architecture = []
        curr_dims = input_dims
        for hl in range(self.num_hidden_layers):
            temp_weights = weight_diist(curr_dims * num_hidden_units).reshape(curr_dims, num_hidden_units)
            self.architecture.append(temp_weights)
            curr_dims = num_hidden_units
        output_weights = weight_diist(num_hidden_units * num_outputs).reshape(num_hidden_units, num_outputs)
        self.architecture.append(output_weights)

        self.matmul_func = np.matmul if not use_torch else torch.matmul

    def sample(self):
        network_input = self.input_dist(self.input_dims)[:, None]
        target = self.forward(network_input)
        return network_input, target

    def forward(self, x):
        output = x
        for hl in range(self.num_hidden_layers):
            output = self.matmul_func(self.architecture[hl].T, output)
        output = self.matmul_func(self.architecture[-1].T, output)
        return output

    def __str__(self):
        network_str = "Network architecture:\n"
        for hl in range(self.num_hidden_layers):
            network_str += "\tLayer {0}:\n\t\tinput size: {1}\toutput size: {2}\tgate function: {3}\n".format(
                hl+1, self.architecture[hl].shape[0],self.architecture[hl].shape[1], self.activation_functions_str
            )
        network_str += "\tOutput Layer:\n\t\tinput size: {0}\toutput size: {1}\tgate function: {2}\n".format(
            self.architecture[-1].shape[0],self.architecture[-1].shape[1], self.activation_functions_str
        )
        return network_str


def main():

    """
    Example of how to use the class FeedForwardTargetGeneratingNetwork
    """
    target_generating_network = FeedForwardTargetGeneratingNetwork(
        input_dims=7,
        num_outputs=3,
        num_hidden_layers=3,
        num_hidden_units=5,
        gate_function="relu",
        input_distribution=distribution(name="normal", parameter_values=(0.0, 2.0)),
        weight_distribution=distribution(name="normal", parameter_values=(0.0, 0.5)),
        use_torch=True
    )

    # print network
    print(target_generating_network)

    # sample ten observations
    sample_size = 10
    for i in range(sample_size):
        obs, target = target_generating_network.sample()
        print("Observation:\n{0}\nTarget:\n{1}\n".format(obs, target))


if __name__ == '__main__':
    main()
