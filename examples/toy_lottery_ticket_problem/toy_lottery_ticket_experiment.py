import torch
import numpy as np
import os
# from project files
from src.definitions import ROOT
from src.experiments import Experiment
from src.function_approximators import DeepNet
from src.util import get_random_seeds, access_dict, distribution
from src.util.neural_networks import layer, get_initialization_function, get_optimizer
from src.problems import FeedForwardTargetGeneratingNetwork
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
        if drop_prob > 0.0:
            architecture.append(layer(type="dropout", parameters=drop_prob, gate=None))

    output_layer = layer(type="linear", parameters=(None, num_outputs), gate=None)
    architecture.append(output_layer)

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
        self.learning_network_dist = exp_params["ln-dist"]
        self.optimizer = exp_params["optimizer"]
        self.stepsize = exp_params["stepsize"]
        self.weight_decay = access_dict(exp_params, key="weight-decay", default=0.0, val_type=float)
        self.drop_prob = access_dict(exp_params, key="dropprob", default=0.0, val_type=float)
        self.lasso_coeff = access_dict(exp_params, key="lasso-coeff", default=0.0, val_type=float)
        # for storing summaries
        self.plot_results = exp_params["plot_results"]
        self.checkpoint = exp_params["checkpoint"]
        self.basic_summaries = access_dict(exp_params, key="basic_summaries", default=False, val_type=bool)
        save_experiment_config_file(self.results_dir, exp_params, run_index)

        """ Define indicator variables and static functions """
        self.l1_reg = (self.lasso_coeff > 0.0)

        """ Network set up """
        self.learning_architecture = get_learning_network_architecture(network_size=self.learning_network_size,
                                                                       depth=self.learning_network_depth,
                                                                       drop_prob=self.drop_prob,
                                                                       num_outputs=self.num_outputs,
                                                                       gate_function=self.learning_network_activation)
        self.net = DeepNet(self.learning_architecture, (1,self.num_inputs), use_bias=False)
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
        new_result_dir = "tgn-size-" + str(self.target_generating_network_size) + "_" + \
                         "tgn-depth-" + str(self.target_generating_network_depth) + "_" + \
                         "tgn-activation-" + self.target_generating_network_activation + "_" + \
                         "tgn-input-dist-" + i_dist[0] + "-" + str(i_dist[1][0]) + "-" + str(i_dist[1][1]) + "_" + \
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
        init_dist = distribution(name=self.learning_network_dist[0], parameter_values=self.learning_network_dist[1:])
        init_function = get_initialization_function(init_dist)
        self.net.apply(init_function)
        # the last layer activation is linear
        if self.learning_network_dist[0] == "kaiming":
            torch.nn.init.kaiming_uniform_(list(self.net.modules())[-1].weight, nonlinearity="linear")

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
            outputs, activations = self.net(net_input.T)

            # compute loss
            current_loss = self.loss_fn(outputs, target.T)

            has_diverged = self.handle_divergence(current_loss)
            if has_diverged: break

            if self.l1_reg:
                current_loss += self.lasso_coeff * sum(torch.abs(p).sum() for p in self.net.parameters())

            # compute backward pass and update parameters
            current_loss.backward()
            optimizer.step()

            # store current summaries and display
            self.store_train_summaries(current_loss.detach())

    def handle_divergence(self, loss):
        """
        Checks if the current loss has diverged and sets results accordingly
        :param loss: (torch tensor) loss function of the current prediction and target
        :return: True if the loss function is nan or infinity
        """
        if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
            curr_cp = (self.current_sample // self.checkpoint) - 1
            self.results_dict["avg_mse_per_cp"][curr_cp:] = torch.inf
            return True
        return False


def main():
    import time

    exp_params = {
        "num-samples": 100000,
        "tgn-size": 64,
        "tgn-depth": 1,
        "tgn-activation": "sigmoid",
        "tgn-input-dist": ("normal", 0.05, 0.01),
        "tgn-weight-dist": ("normal", 1.0, 0.2),
        "num_outputs": 1,
        "num_inputs": 10,
        "ln-size": 128,
        "ln-depth": 1,
        "ln-dist": ("normal", 0.05, 0.02),
        "ln-activation": "sigmoid",
        "optimizer": "sgd",
        "stepsize": 0.05,
        "checkpoint": 100,
        "plot_results": True
    }
    initial_time = time.perf_counter()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lottery_ticket_results")
    exp = TargetGeneratingNetworkExperiment(exp_params, results_dir, 1, True)
    exp.run()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))
    print("Stepsize: {0}".format(exp_params["stepsize"]))


if __name__ == '__main__':
    main()
