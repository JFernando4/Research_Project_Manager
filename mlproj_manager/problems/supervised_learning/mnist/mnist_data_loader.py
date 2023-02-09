"""
Implementation of a data loader for the data set MNIST. This implementation allows the user to select specific
classes to include in or exclude from the data set as well as set specific transformations.
"""
# from third party packages:
import numpy as np
import torch
import torchvision
# from project files:
from mlproj_manager.problems.supervised_learning.abstract_dataset import CustomDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import normalize, preprocess_labels
from mlproj_manager.definitions import MNIST_DATA_PATH


class MnistDataSet(CustomDataSet):

    def __init__(self,
                 root_dir=MNIST_DATA_PATH,
                 train=True,
                 transform=None,
                 classes=None,
                 device=None,
                 image_normalization=None,
                 label_preprocessing=None,
                 use_torch=False):
        """
        :param root_dir (string): path to directory with all the images
        :param train (booL): indicates whether to load the training or testing data set
        :param transform (callable, optional): transform to be applied to each sample
        :param classes (np.array, optional): np array of classes to load. If None, loads all the classes
        :param device (torch.device("cpu") or torch.device("cuda:0") or any variant): device where the data is going
                       to be loaded into. Make sure to have enough memory in your gpu when using "cuda:0" as device.
        :param image_normalization (str, optional): indicates how to normalize the data. options available:
                                                        - None: no normalization
                                                        - centered: subtracts the mean and divides by standard deviation
                                                        - max: divides by 255, the maximum pixel value
        :param label_preprocessing (str, optional): indicates how to preprocess the labels. options available:
                                                        - None: use labels as is (numbers from 0 to 9)
                                                        - one-hot: converts labels to one-hot labels
        :param use_torch (bool, optional): if true, uses torch tensors to represent the data, otherwise, uses np array
        """
        super().__init__(root_dir)
        self.train = train
        self.transform = transform
        self.classes = np.array(classes, np.float32) if classes is not None else np.arange(10)
        assert self.classes.size <= 10, "Cannot select more than 10 classes!"
        self.device = torch.device("cpu") if device is None else device
        self.image_norm_type = image_normalization
        self.label_preprocessing = label_preprocessing
        self.use_torch = use_torch

        self.data = self.load_data()
        self.preprocess_data()
        self.current_data = self.partition_data()

    def load_data(self):
        """
        Load the train or test files using torchvision. The loaded files are not preprocessed.
        :return: dictionary with MNIST dataset
        """
        raw_data_set = torchvision.datasets.MNIST(root=self.root_dir, train=self.train, download=True)
        data = {"data": raw_data_set.data, "labels": raw_data_set.targets}
        return data

    def preprocess_data(self):
        """
        Reshapes the data into the correct dimensions, converts it to the correct type, and normalizes the data if
        specified
        """
        self.data["data"] = self.data["data"].float() if self.use_torch else np.float32(self.data["data"])
        self.data["data"] = normalize(self.data["data"], norm_type=self.image_norm_type,
                                      avg=33.318447,        # average pixel value in mnist
                                      stddev=78.567444,     # standard deviation of pixel values in mnist
                                      max_val=255)          # maximum pixel value in mnist

        new = preprocess_labels(self.data["labels"], preprocessing_type=self.label_preprocessing)
        self.data["labels"] = torch.tensor(new, dtype=torch.float32) if self.use_torch else np.float32(new)

        if self.use_torch:
            self.data["data"] = self.data["data"].to(device=self.device)
            self.data["labels"] = self.data["labels"].to(device=self.device)

    def partition_data(self):
        #TODO: I'll implement this later if necessary, this is just a placeholder in the meantime
        return self.data

    def select_new_partition(self, new_classes):
        #TODO: I'll implement this later if necessary, this is just a placeholder in the meantime
        raise NotImplemented

    def __len__(self):
        """
        :return (int): number of saples in the partitioned data set
        """
        return self.current_data['data'].shape[0]

    def __getitem__(self, idx):
        """
        :param idx (int): valid index from the data set
        :return (dict): the "image" and corresponding "label"
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.current_data["data"][idx]
        label = self.current_data["labels"][idx]
        sample = {"image": image, "label": label}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def set_transformation(self, new_transformation):
        self.transform = new_transformation


def main():

    """ Example of loading and indexing the data set"""
    import matplotlib.pyplot as plt

    # function for plotting a random sample of images
    def plot_four_samples(mnist_data_loader):
        indices = np.random.randint(0, len(mnist_data_loader), size=4)
        fig = plt.figure()
        for i, idx in enumerate(indices):
            sample = mnist_data_loader[idx]

            print("Index: {0}".format(idx))
            print("\tShape [Example, Target]: [{0}, {1}]".format(sample["image"].shape, sample["label"].shape))
            avg_val = np.average(np.float32(sample["image"]))
            std_val = np.std(np.float32(sample["image"]), ddof=1)
            print("\tPixel Value [Average, Standard Deviation]: ({0:.2f}, {1:.2f})".format(avg_val, std_val))
            print("\tMax Pixel Value: {0}".format(np.max(np.float32(sample["image"]))))

            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(idx))
            ax.axis('off')
            plt.imshow(sample["image"].reshape(28,28))
        plt.show()
        plt.close()

    # initialize data set with no normalization
    mnist_data = MnistDataSet(root_dir="./", train=True, classes=None, image_normalization=None,
                              label_preprocessing="one-hot")
    # plot 4 random images
    print("### No Normalization ###")
    plot_four_samples(mnist_data)
    print("\t")

    # initialize data set with "centered" normalization
    mnist_data = MnistDataSet(root_dir="./", train=True, classes=None, image_normalization="centered",
                              label_preprocessing="one-hot")
    # plot 4 random images
    print("### centered ###")
    plot_four_samples(mnist_data)
    print("\t")

    # initialize data set with "max" normalization
    mnist_data = MnistDataSet(root_dir="./", train=True, classes=None, image_normalization="max",
                              label_preprocessing="one-hot")
    print("### max Normalization ###")
    plot_four_samples(mnist_data)
    print("\t")


if __name__ == "__main__":
    main()
