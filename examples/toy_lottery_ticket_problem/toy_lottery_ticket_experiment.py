import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
# from project files
from src.definitions import ROOT, CIFAR_DATA_PATH
from src.experiments import Experiment
from src.function_approximators import DeepNet
from src.util import turn_off_debugging_processes, get_random_seeds, access_dict, distribution
from src.util.neural_networks import layer, add_noise_to_weights, init_weights_kaiming, get_optimizer
from src.problems import FeedForwardTargetGeneratingNetwork
from src.util.data_preprocessing_and_transformations import ToTensor, RandomGaussianNoise, RandomErasing
from src.file_management.file_and_directory_management import save_experiment_config_file
os.chdir(ROOT)


def get_learning_network_architecture(network_size: str, depth: int, drop_prob=0.0, num_outputs=5,
                                      gate_function="relu"):
    """
    Returns a list representing a network architecture according to the given network size
    :param network_size: (int) number of hidden units in each layer
    :param depth: (int) number of layers of the network
    :param drop_prob: (float) dropout probability; must be a float if drop_prob is True
    :param num_outputs: (int) number of output units
    :param gate_function: (int) used for the network
    :return: (list) network architecture
    """

    architecture = []

    for i in range(depth):
        temp_layer = layer(type="linear", parameters=(None, network_size), gate=gate_function)
        architecture.append(temp_layer)

    output_layer = layer(type="linear", parameters=(None, num_outputs), gate=None)
    architecture.append(output_layer)

    if drop_prob > 0.0:
        for i in range(depth):
            temp_idx = 2 * i + 1
            architecture.insert(temp_idx, layer(type="dropout", parameters=drop_prob, gate=None))

    return architecture


class TargetGeneratingNetworkExperiment(Experiment):

    """
    Simple experiment where the goal is to produce the same output as a static network
    """

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        """ For reproducibility """
        random_seeds = get_random_seeds()
        """ Experiment parameters """
        # for defining the learning problem
        self.num_samples = exp_params["num-samples"]
        self.target_generating_network_size = exp_params["tgn-size"]
        self.target_generating_network_depth = exp_params["tgn-depth"]
        self.target_generating_network_activation = exp_params["tgn-activation"]
        i_dist = exp_params["tgn-input-dist"]
        self.target_generating_network_input_dist = distribution(name=i_dist[0], parameter_values=i_dist[1:])
        w_dist = exp_params["tgn-weight-dist"]
        self.target_generating_network_weight_dist = distribution(name=w_dist[0], parameter_values=w_dist[1:])
        self.num_outputs = exp_params["num_outputs"]
        self.num_inputs = exp_params["num_inputs"]
        self.update_results_dir()
        # for defining the learning algorithm
        self.learning_network_size = exp_params["ln-size"]
        self.learning_network_depth = exp_params["ln-depth"]
        self.learning_network_activation = exp_params["ln-activation"]
        self.optimizer = exp_params["optimizer"]
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = access_dict(exp_params, key="weight-decay", default=0.0, val_type=float)
        self.drop_prob = access_dict(exp_params, key="dropprob", default=0.0, val_type=float)
        self.lasso_coeff = access_dict(exp_params, key="lasso-coeff", default=0.0, val_type=float)
        # for storing summaries
        self.plot_results = exp_params["plot_results"]
        self.checkpoint = exp_params["checkpoint"]
        self.basic_summaries = access_dict(exp_params, key="basic_summaries", default=False, val_type=bool)
        save_experiment_config_file(results_dir, exp_params, run_index)

        """ Define indicator variables and static functions """
        self.l1_reg = (self.lasso_coeff > 0.0)

        """ Network set up """
        self.learning_architecture = get_learning_network_architecture(network_size=self.learning_network_size,
                                                                       depth=self.learning_network_depth,
                                                                       drop_prob=self.drop_prob,
                                                                       num_outputs=self.num_outputs,
                                                                       gate_function=self.learning_network_activation)
        self.net = DeepNet(self.learning_architecture, (1,self.num_inputs))
        self.scripted_forward_pass = None
        self.loss_fn = torch.nn.MSELoss(reduction="mean")

        """ For summaries """
        self.random_seed = random_seeds[self.run_index]
        self.current_sample = 0
        assert self.num_samples % self.checkpoint == 0
        num_cps = self.num_samples // self.checkpoint

        self.results_dict["avg_mse_per_cp"] = torch.zeros(num_cps, dtype=torch.float32)
        self.running_mse = None
        self.reset_running_estimates()

    # ---- For printing and plotting summaries ---- #
    def print_training_progress(self, curr_cp):
        print_message = "\tObservation number: {0}\tLoss: {1:.4f}"
        self._print(print_message.format(self.current_sample, self.results_dict["avg_mse_per_cp"][curr_cp]))

    def plot(self):
        if not self.plot_results:
            return
        import matplotlib.pyplot as plt
        np_results = self.results_dict["avg_mse_per_cp"].cpu().numpy()
        plt.plot(np.arange(np_results.size), np_results)
        plt.show()
        plt.close()

    # --- For storing and computing summaries --- #
    def update_results_dir(self):
        """
        Creates a new result directory according to the information about the target generating network
        """
        i_dist = tuple(self.target_generating_network_input_dist)
        w_dist = tuple(self.target_generating_network_weight_dist)
        new_result_dir = "tgn-size-" + str(self.target_generating_network_size) + "_" +\
                         "tgn-depth-" + str(self.target_generating_network_depth) + "_" +\
                         "tgn-activation-" + self.target_generating_network_activation + "_" +\
                         "tgn-input-dist-" + i_dist[0] + "-" + str(i_dist[1][0]) + "-" + str(i_dist[1][1]) + "_" +\
                         "tng-weight_dist-" + w_dist[0] + "-" + str(w_dist[1][0]) + "-" + str(w_dist[1][1]) + "_" + \
                         "num-inputs-" + str(self.num_inputs) + "_num-outputs-" + str(self.num_outputs)
        new_dir_path = os.path.join(self.results_dir, new_result_dir)
        os.makedirs(new_dir_path, exist_ok=True)
        self.results_dir = new_dir_path

    def reset_running_estimates(self):
        self.running_mse = torch.tensor(0.0, dtype=torch.float32)

    def store_train_summaries(self, curr_loss):

        self.running_mse += curr_loss

        if self.current_sample % self.checkpoint == 0:
            curr_cp = (self.current_sample // self.checkpoint) - 1
            self.results_dict["avg_mse_per_cp"][curr_cp] += self.running_mse
            self.reset_running_estimates()
            self.print_training_progress(curr_cp)

    # ---- For running the experiment ---- #
    def run(self):

        # set random seeds
        torch.random.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # intialize target generating network
        target_generating_network = FeedForwardTargetGeneratingNetwork(
            input_dims=self.num_inputs,
            num_outputs=self.num_outputs,
            num_hidden_layers=self.target_generating_network_depth,
            num_hidden_units=self.target_generating_network_size,
            gate_function=self.target_generating_network_activation,
            input_distribution=self.target_generating_network_input_dist,
            weight_distribution=self.target_generating_network_weight_dist,
            use_torch=True
        )

        # initialize learning network
        self.net.apply(lambda x: init_weights_kaiming(x, normal=False, nonlinearity="relu"))
        # the last layer activation is linear
        torch.nn.init.kaiming_uniform_(list(self.net.modules())[-1].weight, nonlinearity="linear")

        # dropout simply doesn't work with jit.trace, or at least I couldn't get it to work properly
        if self.drop_prob == 0.0:
            dummy_sample = torch.zeros((1, self.num_inputs))
            self.scripted_forward_pass = torch.jit.trace(self.net, dummy_sample)
        else:
            self.scripted_forward_pass = self.net

        optimizer = get_optimizer(self.optimizer, self.net.parameters(), stepsize=self.stepsize,
                                  weight_decay=self.weight_decay)

        # train network
        self.train(optimizer, target_generating_network)

        # plot results
        self.plot()

    def train(self, optimizer: torch.optim.Optimizer, tg_network: FeedForwardTargetGeneratingNetwork):

        for t in range(self.num_samples):
            self.current_sample += 1

            # get new sample
            net_input, target = tg_network.sample()

            # reset gradients
            for param in self.net.parameters(): param.grad = None

            # forward pass
            outputs, activations = self.scripted_forward_pass(net_input.T)

            # compute loss
            current_loss = self.loss_fn(outputs, target.T)
            if self.l1_reg:
                current_loss += self.lasso_coeff * sum(torch.abs(p).sum() for p in self.net.parameters())

            # compute backward pass and update parameters
            current_loss.backward()
            optimizer.step()

            # store current summaries and display
            self.store_train_summaries(current_loss.detach())


def main():
    import time

    exp_params = {
        "num-samples": 20000,
        "tgn-size": 256,
        "tgn-depth": 3,
        "tgn-activation": "relu",
        "tgn-input-dist": ("normal", 0, 0.2),
        "tgn-weight-dist": ("normal", 0, 0.5),
        "num_outputs": 5,
        "num_inputs": 10,
        "ln-size":  256,
        "ln-depth": 3,
        "ln-activation": "relu",
        "optimizer": "sgd",
        "stepsize": 0.01,
        "checkpoint": 100,
        "plot_results": True
    }

    initial_time = time.perf_counter()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lottery_ticket_results")
    exp = TargetGeneratingNetworkExperiment(exp_params, results_dir, 1, True)
    exp.run()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == '__main__':
    main()
