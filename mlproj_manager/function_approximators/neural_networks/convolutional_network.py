
# from third party packages:
import torch
import torch.nn as nn
import numpy as np
# from project files:
from mlproj_manager.util.neural_networks.network_architecture import layer, get_activation, get_conv_layer_output_dims


class DeepNet(nn.Module):
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
        super(DeepNet, self).__init__()

        # check that each layer has the correct format
        assert all(isinstance(a_layer, layer) for a_layer in architecture)

        self.architecture = architecture
        self.activations = []
        for a_layer in architecture:
            self.activations.append(get_activation(a_layer.gate))

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

    def forward(self, obs, return_activations=True):
        x = obs
        activations = ()

        for i, module in enumerate(self.network_module_list):
            # apply layer
            x = module(x)
            # apply gate function
            x = self.activations[i](x)
            # apply mask: this is for shrinking or masking out certain features
            layer_type = self.architecture[i].type
            conv_or_feedforward = layer_type in ["conv2d", "linear"]
            if return_activations and conv_or_feedforward:
                activations += (x, )

        if return_activations:
            return x, activations[:-1]
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

    network = DeepNet(architecture, (32, 32))
    network.apply(lambda z: xavier_init_weights(z, normal=True))

    image = torch.normal(0, 0.5, (3, 32, 32))
    output = network.forward(image[None, :], return_activations=False)
    print("Output: \n\t", output, "\nSum: ", torch.sum(output))

    # loading and the data set
    print("\nLoading CIFAR-100 daata set...")
    cifar_data_set = CifarDataSet(train=True, root_dir=os.path.join(ROOT, "problems", "supervised_learning", "cifar"),
                                  transform=ToTensor(), classes=np.arange(num_classes, dtype=np.int32),
                                  image_normalization="minus-one-to-one", label_preprocessing="one-hot",
                                  cifar_type=100)
    trainloader = torch.utils.data.DataLoader(cifar_data_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # setting up optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=stepsize)

    # loss record
    avg_loss_per_checkpoint = []
    accuracy_per_checkpoint = []

    # start training
    for epoch in range(num_epochs):
        print("Epoch number: {0}".format(epoch + 1))

        running_loss = 0.0
        running_accuracy = 0.0
        for i, sample in enumerate(trainloader, 0):
            # get batch of images and labels
            current_batch += 1
            images = sample["image"]
            labels = sample["label"]

            # zero the gradients
            optimizer.zero_grad()

            # forward + backward passes + update
            outputs = network.forward(images, return_activations=False)
            current_loss = loss_function(outputs, labels)
            current_loss.backward()
            optimizer.step()

            # print and store stats
            running_loss += current_loss.item() / checkpoint
            running_accuracy += torch.mean((torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).double()).item() / checkpoint
            if current_batch % checkpoint == 0:
                print("\tBatch Number: {0}\tLoss: {1:.2f}\tAccuracy: {2:.2f}".format(current_batch, running_loss, running_accuracy))
                avg_loss_per_checkpoint.append(running_loss)
                accuracy_per_checkpoint.append(running_accuracy)
                running_loss = 0.0
                running_accuracy = 0.0

        new_transformation = transforms.Compose([
            ToTensor(),
            RandomGaussianNoise(mean=0, stddev=0.4),
            RandomErasing(scale=(0.24, 0.32), ratio=(1, 2))
        ])
        cifar_data_set.set_transformation(new_transformation)

    print("Finished training...")

    plt.plot(np.arange(len(avg_loss_per_checkpoint)), avg_loss_per_checkpoint)
    plt.show()
    plt.plot()

    plt.plot(np.arange(len(accuracy_per_checkpoint)), accuracy_per_checkpoint)
    plt.show()
    plt.plot()

    new_transformation = transforms.Compose([
        ToTensor(),
        RandomGaussianNoise(mean=0, stddev=0.4),
        RandomErasing(scale=(0.24, 0.32), ratio=(1, 2))
    ])
    cifar_test_data = CifarDataSet(train="test", root_dir=os.path.join(ROOT, "problems", "supervised_learning", "cifar"),
                                   transform=new_transformation, classes=np.arange(num_classes, dtype=np.int32),
                                   image_normalization="minus-one-to-one", label_preprocessing="one-hot",
                                   cifar_type=100, use_torch=True)
    test_data = next(iter(torch.utils.data.DataLoader(cifar_test_data, batch_size=2500, shuffle=True, num_workers=2)))
    outputs = network.forward(test_data["image"], return_activations=False)
    labels = test_data["label"]
    test_accuracy = torch.mean((torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).double()).item()
    print("The test accuracy is: {0:.2f}".format(test_accuracy))


if __name__ == "__main__":
    main()
