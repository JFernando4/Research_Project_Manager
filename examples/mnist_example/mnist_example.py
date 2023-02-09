import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
import time
# from project files
from mlproj_manager.definitions import ROOT, MNIST_DATA_PATH
from mlproj_manager.experiments import Experiment
from mlproj_manager.function_approximators import DeepNet
from mlproj_manager.util import turn_off_debugging_processes, get_random_seeds, access_dict
from mlproj_manager.util.neural_networks import layer, init_weights_kaiming, get_optimizer
from mlproj_manager.problems import MnistDataSet
os.chdir(ROOT)


class MNISTExperiment(Experiment):

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
        self.training_epochs = exp_params["training_epochs"]
        self.batch_size = exp_params['batch_size']
        self.num_units_per_layer = access_dict(exp_params, key="num_units_per_layer", default="large", val_type=int)
        self.num_layers = access_dict(exp_params, key="num_layers", default="large", val_type=int)
        self.gate_function = access_dict(exp_params, key="gate_function", default="relu",
                                         choices=["relu", "sigmoid", "tanh", None])
        self.data_path = exp_params['data_path']

        """ Training constants """
        self.num_classes = 10
        self.image_dims = (28, 28)
        self.num_images_per_epoch = 60000

        """ Network set up """
        # initialize network architecture
        architecture = []
        for _ in range(self.num_layers):
            architecture.append(layer(type="linear", parameters=(None, self.num_units_per_layer), gate="relu"))
        architecture.append(layer(type="linear", parameters=(None, self.num_classes), gate=None))
        # initialize network
        self.net = DeepNet(architecture, image_dims=self.image_dims)
        # initialize network weights
        init_func = lambda m: init_weights_kaiming(m, nonlinearity=self.gate_function, normal=True)
        self.net.apply(init_func)
        # initialize optimizer
        self.optim = get_optimizer(self.optimizer, self.net.parameters(), stepsize=self.stepsize)
        # define loss function
        self.loss = torch.nn.CrossEntropyLoss(reduction="mean")
        # move network to device
        self.net.to(self.device)

        """ For summaries and reproducibility """
        self.random_seed = random_seeds[self.run_index]
        # summaries
        for data_type in ["test", "avg_train"]:
            for summary_type in ["acc", "loss"]:
                name = data_type + "_" + summary_type + "_per_epoch"
                self.results_dict[name] = torch.zeros(self.training_epochs, device=self.device, dtype=torch.float32)

    # -------------------- For manipulating data -------------------- #
    def get_data(self, train=True, return_data_loader=False):
        """ Loads MNIST data set """
        mnist_data = MnistDataSet(root_dir=self.data_path,
                                  train=train,
                                  device=self.device,
                                  image_normalization="max",
                                  label_preprocessing="one-hot",
                                  use_torch=True)
        if return_data_loader:
            dataloader = DataLoader(mnist_data, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                    pin_memory=False)
            return mnist_data, dataloader

        return mnist_data

    # -------------------- For storing summaries -------------------- #
    def store_summaries(self, avg_train_loss: torch.Tensor, test_loss: torch.Tensor, avg_train_acc: torch.Tensor,
                        test_acc: torch.Tensor, idx: int = 0):

        self.results_dict["test_acc_per_epoch"][idx] += test_acc
        self.results_dict["test_loss_per_epoch"][idx] += test_loss
        self.results_dict["avg_train_acc_per_epoch"][idx] += avg_train_acc
        self.results_dict["avg_train_loss_per_epoch"][idx] += avg_train_loss

    # -------------------- For running the experiment -------------------- #
    def train_one_epoch(self, mnist_dataloader: DataLoader):

        loss_per_batch = torch.zeros(len(mnist_dataloader), dtype=torch.float32, device=self.device)
        acc_per_batch = torch.zeros(len(mnist_dataloader), dtype=torch.float32, device=self.device)
        for i, sample in enumerate(mnist_dataloader):
            image = sample["image"].reshape(self.batch_size, np.prod(self.image_dims))
            label = sample["label"]

            for param in self.net.parameters(): param.grad = None   # apparently faster than optim.zero_grad()

            outputs = self.net.forward(image, return_activations=False)
            current_loss = self.loss(outputs, label)
            current_loss.backward()
            self.optim.step()

            loss_per_batch[i] += current_loss.detach()
            accuracy = torch.mean((outputs.argmax(axis=1) == label.argmax(axis=1)).to(torch.float32))
            acc_per_batch[i] += accuracy.detach()

        return loss_per_batch, acc_per_batch

    def evaluate_network(self, test_data: MnistDataSet):
        with torch.no_grad():
            test_outputs = self.net.forward(test_data[:]["image"].reshape(-1, np.prod(self.image_dims)),
                                            return_activations=False)
            test_labels = test_data[:]["label"]

            loss = self.loss(test_outputs, test_labels)
            acc = torch.mean((test_outputs.argmax(axis=1) == test_labels.argmax(axis=1)).to(torch.float32))

        return loss, acc

    def train(self, mnist_data_loader: DataLoader, test_data: MnistDataSet):
        self._print("Training network...")

        for e in range(self.training_epochs):
            self._print("\tEpoch number: {0}".format(e + 1))
            loss, acc = self.train_one_epoch(mnist_dataloader=mnist_data_loader)
            test_loss, test_accuracy = self.evaluate_network(test_data)

            self.store_summaries(avg_train_loss=torch.mean(loss),
                                 avg_train_acc=torch.mean(acc),
                                 test_loss=test_loss,
                                 test_acc=test_accuracy,
                                 idx=e)

            self._print("\t\tTest accuracy: {0:.4f}".format(test_accuracy))

    def run(self):

        # load data
        training_data, training_dataloader = self.get_data(train=True, return_data_loader=True)
        test_data = self.get_data(train=False, return_data_loader=False)

        # train network
        self.train(mnist_data_loader=training_dataloader, test_data=test_data)

        # save summaries to memory
        self.store_results()


def main():
    """
    This is a quick demonstration of how to run the experiments. For a more systematic run, use the mlproj_manager
    scheduler.
    """
    file_path = os.path.dirname(os.path.abspath(__file__))
    experiment_parameters = {
        "optimizer": "sgd",
        "stepsize": 0.1,
        "training_epochs": 10,
        "batch_size": 32,
        "num_units_per_layer": 100,
        "num_layers": 1,
        "gate_function": "relu",
        "data_path": MNIST_DATA_PATH,
    }

    initial_time = time.perf_counter()
    exp = MNISTExperiment(experiment_parameters,
                          results_dir=os.path.join(file_path, "results"),
                          run_index=0,
                          verbose=True)
    exp.run()
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
