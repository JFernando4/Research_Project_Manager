import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
# from project files
from mlproj_manager.definitions import ROOT, CIFAR_DATA_PATH
from mlproj_manager.experiments import Experiment
from mlproj_manager.function_approximators import DeepNet
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.neural_networks import layer, add_noise_to_weights, init_weights_kaiming, get_optimizer, scale_weights
from mlproj_manager.problems import CifarDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import ToTensor, RandomGaussianNoise, RandomErasing
from mlproj_manager.file_management.file_and_directory_management import save_experiment_config_file
os.chdir(ROOT)


def get_cifar_architecture(network_size: str, num_classes: int, drop_prob=0.0):
    """
    Returns a list representing a network architecture according to the given network size
    :param network_size: (str) "small", "medium", "large", "extra-large"
    :param num_classes: (int) number of classes in cifar partition
    :param drop_prob: (float) dropout probability; must be a float if drop_prob is True
    :return: (list) network architecture
    """
    units_first = {"small": 16, "medium": 32, "large": 64, "extra-large": 256}[network_size]
    units_second = {"small": 8, "medium": 16, "large": 32, "extra-large": 128}[network_size]
    units_feedforward = {"small": 256, "medium": 512, "large": 1024, "extra-large": 4096}[network_size]

    architecture = [
        layer(type="conv2d",    parameters=(3, units_first, (3,3), (1,1)),              gate="relu"),   # conv 1
        layer(type="maxpool",   parameters=((2,2), (2,2)),                              gate=None),     # max pool 1
        layer(type="conv2d",    parameters=(units_first, units_second, (3,3), (1,1)),   gate="relu"),   # conv 2
        layer(type="maxpool",   parameters=((2,2), (1,1)),                              gate=None),     # max pool 2
        layer(type="flatten",   parameters=(),                                          gate=None),     # flatten
        layer(type="linear",    parameters=(None, units_feedforward),                   gate='relu'),   # feed forward 1
        layer(type="linear",    parameters=(None, num_classes),                         gate=None)      # output
    ]

    if drop_prob > 0.0:
        assert drop_prob is not None
        architecture.insert(2, layer(type="dropout", parameters=drop_prob, gate=None))
        architecture.insert(5, layer(type="dropout", parameters=drop_prob, gate=None))
        architecture.insert(8, layer(type="dropout", parameters=drop_prob, gate=None))

    return architecture


class NonStationaryCifarExperiment(Experiment):

    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose=True):
        super().__init__(exp_params, results_dir, run_index, verbose)

        # set debugging options for pytorch
        debug = access_dict(exp_params, key="debug", default=True, val_type=bool)
        turn_off_debugging_processes(debug)

        # define torch device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        """ For reproducibility """
        random_seeds = get_random_seeds()

        """ Experiment parameters """
        self.optimizer = exp_params["optimizer"]
        self.stepsize = exp_params["stepsize"]
        self.plot_results = exp_params["plot_results"]
        self.epochs_per_task = exp_params["epochs_per_task"]
        self.checkpoint = exp_params["checkpoint"]
        self.data_path = exp_params['data_path']
        self.batch_size = exp_params['batch_size']
        self.weight_decay = access_dict(exp_params, key="weight_decay", default=0.0, val_type=float)
        self.drop_prob = access_dict(exp_params, key="dropprob", default=0.0, val_type=float)
        self.lasso_coeff = access_dict(exp_params, key="lasso_coeff", default=0.0, val_type=float)
        self.scale_factor = access_dict(exp_params, key="scale_factor", default=1.0, val_type=float)
        self.epoch_noise_std = access_dict(exp_params, key="epoch_noise_std", default=0.0, val_type=float)
        self.iter_noise_std = access_dict(exp_params, key="iter_noise_std", default=0.0, val_type=float)
        self.network_size = access_dict(exp_params, key="network_size", default="large", val_type=str)
        self.basic_summaries = access_dict(exp_params, key="basic_summaries", default=False, val_type=bool)
        self.image_norm_type = access_dict(exp_params, key="image_norm_type", default="minus-one-to-one", val_type=str)
        save_experiment_config_file(results_dir, exp_params, run_index)

        """ Define indicator variables and static functions """
        self.l1_reg = (self.lasso_coeff > 0.0)
        self.scale = (self.scale_factor < 1.0)
        self.scale_func = lambda x: scale_weights(x, self.scale_factor)
        self.perturb = (self.epoch_noise_std > 0.0)
        self.epoch_noise_func = lambda x: add_noise_to_weights(x, stddev=self.epoch_noise_std)
        self.iteration_noise = (self.iter_noise_std > 0.0)
        self.iteration_noise_func = lambda x: add_noise_to_weights(x, stddev=self.iter_noise_std)

        """ Training constants """
        self.num_classes = 5
        self.image_dims = (32,32)
        self.num_images_per_epoch = 2500

        """ Network set up """
        self.architecture = get_cifar_architecture(self.network_size, num_classes=self.num_classes,
                                                   drop_prob=self.drop_prob)
        self.net = DeepNet(self.architecture, self.image_dims)
        self.scripted_forward_pass = None
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # move network parameters and other variables to the device
        self.scale_factor = torch.tensor(self.scale_factor, dtype=torch.float32, device=self.device)
        self.epoch_noise_std = torch.tensor(self.epoch_noise_std, dtype=torch.float32, device=self.device)
        self.tensor_batch_size = torch.tensor(self.batch_size, dtype=torch.float32, device=self.device)
        self.tensor_cp = torch.tensor(self.checkpoint, dtype=torch.float32, device=self.device)
        self.net.to(self.device)

        """ For Summaries """
        self.random_seed = random_seeds[self.run_index]
        self.test_data_size = 500
        self.current_epoch = 0
        self.current_obs = 0
        # train summaries
        num_partitions = 100 // self.num_classes
        num_cps = (self.num_images_per_epoch // (self.checkpoint * self.batch_size)) * self.epochs_per_task * num_partitions
        self.results_dict["avg_accuracy_per_cp"] = torch.zeros(num_cps, device=self.device, dtype=torch.float32)
        self.results_dict["avg_loss_per_cp"] = torch.zeros(num_cps, device=self.device, dtype=torch.float32)
        self.results_dict["test_accuracy_per_cp"] = torch.zeros(num_cps, device=self.device, dtype=torch.float32)

        self.running_loss = self.running_accuracy = None
        self.reset_running_estimates()

    # ---- For printing and plotting summaries ---- #
    def plot(self):
        if not self.plot_results:
            return
        import matplotlib.pyplot as plt

        N = 10
        results_to_plot = [self.results_dict["avg_accuracy_per_cp"], self.results_dict["avg_loss_per_cp"],
                           self.results_dict["test_accuracy_per_cp"]]
        for res in results_to_plot:
            moving_average = np.convolve(res.cpu(), np.ones(N) / N, mode="valid")
            plt.plot(np.arange(moving_average.size), moving_average); plt.show(), plt.close()

    def print_training_progress(self, curr_cp):
        print_message = "\tObservation Number: {0}\tLoss: {1:.2f}\tAccuracy: {2:.2f}"
        self._print(print_message.format(self.current_obs,
                                         self.results_dict["avg_loss_per_cp"][curr_cp],
                                         self.results_dict["avg_accuracy_per_cp"][curr_cp]))

    # ---- For storing and computing summaries ---- #
    def reset_running_estimates(self):
        self.running_loss = torch.tensor(0.0,device=self.device,dtype=torch.float32)
        self.running_accuracy = torch.tensor(0,device=self.device,dtype=self.results_dict["avg_accuracy_per_cp"].dtype)

    def store_train_summary(self, predictions, labels, loss):
        accuracy = torch.mean((torch.argmax(predictions, 1) == torch.argmax(labels, 1)).to(torch.float32))
        curr_cp = (self.current_obs // self.checkpoint) - 1

        self.running_loss += loss
        self.running_accuracy += accuracy

        if self.current_obs % self.checkpoint == 0:
            self.results_dict["avg_accuracy_per_cp"][curr_cp] += self.running_accuracy / self.tensor_cp
            self.results_dict["avg_loss_per_cp"][curr_cp] += self.running_loss / self.tensor_cp
            # reset estimates and display
            self.reset_running_estimates()
            self.print_training_progress(curr_cp)

    def store_test_summary(self, test_data):
        if self.basic_summaries: return     # skip this if only storing basic summaries

        if self.current_obs % self.checkpoint == 0:
            curr_cp = (self.current_obs // self.checkpoint) - 1
            self.net.eval()
            outputs = self.net.forward(test_data["image"], return_activations=False)
            labels = test_data["label"]
            test_accuracy = torch.mean((torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).double()).item()
            self.results_dict["test_accuracy_per_cp"][curr_cp] += test_accuracy
            self.net.train()

    # ---- For manipulating the data ---- #
    def get_data_set(self, train=True):
        mnist_data_set = CifarDataSet(root_dir=self.data_path, train=train, transform=ToTensor(), device=self.device,
                                      image_normalization=self.image_norm_type, label_preprocessing="one-hot",
                                      use_torch=True)
        batch_size = self.batch_size if train else self.test_data_size
        return mnist_data_set, DataLoader(mnist_data_set, batch_size, shuffle=True, num_workers=0, pin_memory=False)

    # ---- For running the experiment ---- #
    def run(self):

        # set random seeds
        torch.random.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        # initialize the network
        self.net.apply(lambda x: init_weights_kaiming(x, normal=False, nonlinearity="relu"))
        # the last layer activation is linear
        torch.nn.init.kaiming_uniform_(list(self.net.modules())[-1].weight, nonlinearity="linear")

        # dropout simply doesn't work with jit.trace, or at least I couldn't get it to work properly
        if self.drop_prob == 0.0:
            dummy_sample = torch.zeros((1, 3) + self.image_dims, device=self.device)
            self.scripted_forward_pass = torch.jit.trace(self.net, dummy_sample)
        else:
            self.scripted_forward_pass = self.net

        # generate optimizer and load data
        optimizer = get_optimizer(self.optimizer, self.net.parameters(), stepsize=self.stepsize,
                                  weight_decay=self.weight_decay)
        train_data_set = self.get_data_set(train=True)
        test_data_set = self.get_data_set(train=False)

        # train network
        self.train(optimizer, train_data_set, test_data_set)
        # plot results
        self.plot()

    def train(self, optimizer: torch.optim.Optimizer, train_data_set: tuple, test_data_set: tuple):
        # set up data set
        train_data, train_data_loader = train_data_set
        test_data, test_data_loader = test_data_set

        # get partitions
        classes = np.arange(100)
        np.random.shuffle(classes)
        partitions = np.split(classes, 100 // self.num_classes)

        for part in partitions:
            train_data.select_new_partition(part)
            train_data.set_transformation(ToTensor())
            test_data.select_new_partition(part)
            temp_test_data = next(iter(test_data_loader))

            # train network
            for epoch in range(self.epochs_per_task):
                self.current_epoch += 1
                self._print("Epoch number: {0}".format(self.current_epoch))

                # process each batch
                for i, sample in enumerate(train_data_loader):
                    self.current_obs += 1
                    image = sample["image"]
                    label = sample["label"]
                    # reset gradients
                    for param in self.net.parameters(): param.grad = None
                    # forward + backward passes + optimizer step
                    outputs, activations = self.scripted_forward_pass(image)
                    current_loss = self.loss_fn(outputs, label)
                    if self.l1_reg:
                        current_loss += self.lasso_coeff * sum(torch.abs(p).sum() for p in self.net.parameters())
                    current_loss.backward()
                    optimizer.step()
                    # store current summaries and display
                    self.store_train_summary(outputs.detach(), label, current_loss.detach())
                    self.store_test_summary(temp_test_data)
                    # add parameter noise
                    if self.iteration_noise: self.net.apply(self.iteration_noise_func)

                if self.scale: self.net.apply(self.scale_func)             # shrink
                if self.perturb: self.net.apply(self.epoch_noise_func)    # perturb

                # get a new random permutation and update the train and test data
                new_transformation = transforms.Compose([
                    ToTensor(),
                    RandomGaussianNoise(mean=0, stddev=np.random.normal(0.5, 0.1)),
                    RandomErasing(scale=(np.random.normal(0.2, 0.025), np.random.normal(0.4, 0.025)),
                                  ratio=(np.random.uniform(0.1, 1), np.random.uniform(1.1, 2)))
                ])
                train_data.set_transformation(new_transformation)


def main():
    exp_params = {
        "optimizer": "adam",
        "stepsize": 0.01,
        "plot_results": True,
        "epochs": 200,
        "checkpoint": 5,
        "data_path": CIFAR_DATA_PATH,
        "batch_size": 4,
        "network_size": "large"
    }
    import time

    initial_time = time.perf_counter()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "non_stationary_cifar_results")
    exp = NonStationaryCifarExperiment(exp_params, results_dir, 1, True)
    exp.run()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == '__main__':
    main()
