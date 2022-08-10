"""
Implementation of a data loader for the data set CIFAR-100. This implementation allows the user to select specific
classes to include in or exclude from the data set.
"""
# from third party packages:
import torch
import numpy as np
import torchvision
# from project files:
from mlproj_manager.problems.supervised_learning.abstract_dataset import CustomDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import normalize, preprocess_labels
from mlproj_manager.definitions import CIFAR_DATA_PATH


class CifarDataSet(CustomDataSet):
    """ CIFAR dataset."""

    def __init__(self,
                 root_dir=CIFAR_DATA_PATH,
                 train=True,
                 transform=None,
                 cifar_type=100,
                 classes=None,
                 device=None,
                 image_normalization=None,
                 label_preprocessing=None,
                 use_torch=False,
                 flatten=False):
        """
        :param root_dir (string): path to directory with all the images
        :param train (booL): indicates whether to load the training or testing data set
        :param transform (callable, optional): transform to be applied to each sample
        :param cifar_type (int): indicates whether to use cifar10 (cifar_type=10) or cifar100 (cifar_type=100) data set
        :param classes (np.array, optional): np array of classes to load. If None, loads all the classes
        :param device (torch.device("cpu") or torch.device("cuda:0") or any variant): device where the data is going
                       to be loaded into. Make sure to have enough memory in your gpu when using "cuda:0" as device.
        :param image_normalization (str, optional): indicates how to normalize the data. options available:
                                                        - None: no normalization
                                                        - centered: subtracts the mean and divides by standard deviation
                                                        - max: divides by 255, the maximum pixel value
                                                        - minus-one-to-one: divides by the maximum of the data and
                                                          scales to range(-1,1)
        :param label_preprocessing (str, optional): indicates how to preprocess the labels. options available:
                                                        - None: use labels as is (numbers from 0 to 9)
                                                        - one-hot: converts labels to one-hot labels
        :param use_torch (bool, optional): if true, uses torch tensors to represent the data, otherwise, uses np array
        :param flatten: whether to flatten each image
        """
        super().__init__(root_dir)
        self.train = train
        self.transform = transform
        self.cifar_type = cifar_type
        self.classes = np.array(classes, np.int32) if classes is not None else np.arange(self.cifar_type)
        assert self.classes.size <= 100
        self.device = torch.device("cpu") if device is None else device
        self.image_norm_type = image_normalization
        self.label_preprocessing = label_preprocessing
        self.use_torch = use_torch
        self.flatten = flatten

        self.data = self.load_data()
        self.integer_labels = self.data["labels"]
        self.preprocess_data()
        self.current_data = self.partition_data()

    def load_data(self):
        """
        Load the train or test files using torchvision. The loaded files are not preprocessed. It downloads the data if
        necessary and stores it in "./src/problems/supervised_learning/cifar/". It creates several files, see
        https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10 for mor info
        :return: dictionary with keys "data" and "labels"
        """
        # Note: the shape of CIFAR10 and 100 is (num_samples, 32, 32, 3)
        if self.cifar_type == 10:
            raw_data_set = torchvision.datasets.CIFAR10(root=self.root_dir, train=self.train, download=True)
        elif self.cifar_type == 100:
            raw_data_set = torchvision.datasets.CIFAR100(root=self.root_dir, train=self.train, download=True)
        else:
            raise ValueError("Invalid CIFAR type: {0}. Choose either 100 or 10".format(self.cifar_type))
        data = {"data": raw_data_set.data, "labels": raw_data_set.targets}
        return data

    def preprocess_data(self):
        """
        Reshapes the data into the correct dimensions, converts it to the correct type, and normalizes the data if
        if specified
        :return: None
        """
        if self.flatten:
            self.data["data"] = self.data["data"].reshape(self.data["data"].shape[0], -1)

        self.data["data"] = torch.tensor(self.data["data"], dtype=torch.float32) if self.use_torch \
            else np.float32(self.data["data"])
        avg_px_val = 120.70756512369792 if self.cifar_type == 10 else 121.936059453125
        stddev_px_val = 64.15007609994316 if self.cifar_type == 10 else 68.38895681157001
        self.data["data"] = normalize(self.data["data"], norm_type=self.image_norm_type,
                                      avg=avg_px_val, stddev=stddev_px_val,
                                      max_val=255)  # max value of pixels in CIFAR10 and CIFAR100 images

        new = preprocess_labels(self.data["labels"], preprocessing_type=self.label_preprocessing)
        self.data["labels"] = torch.tensor(new, dtype=torch.float32) if self.use_torch else np.float32(new)

        if self.use_torch:
            self.data["data"] = self.data["data"].to(device=self.device)
            self.data["labels"] = self.data["labels"].to(device=self.device)

    def partition_data(self):
        """
        partitions the CIFAR data sets according to the classes in self.classes
        :return (dict): contains the samples and labels corresponding to the specified classes
        """
        current_data = {}

        # get indices corresponding to the classes
        correct_rows = np.ones(self.data['data'].shape[0], dtype=bool)
        if self.classes is not None:
            correct_rows = np.in1d(self.integer_labels, self.classes)

        # store samples and labels in the new dictionary
        current_data['data'] = self.data['data'][correct_rows, :, :, :]
        if self.label_preprocessing == "one-hot":
            current_data['labels'] = self.data['labels'][correct_rows][:,self.classes]
        else:
            current_data['labels'] = self.data['labels'][correct_rows]
        return current_data

    def select_new_partition(self, new_classes):
        """
        Generates a new partition of the CIFAR-100 data set based on the classes in new_classes
        :param new_classes: (list) the classes cof the new partition
        :return: None
        """
        self.classes = np.array(new_classes, dtype=np.int32)
        self.current_data = self.partition_data()

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

        image = self.current_data['data'][idx]
        label = self.current_data['labels'][idx]    # one hot labels
        # label = self.current_data['fine_labels'][idx]     # regular labels
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def set_transformation(self, new_transformation):
        self.transform = new_transformation


def main():

    """ Testing loading and indexing the data set"""
    import matplotlib.pyplot as plt

    # function for plotting a random sample of images
    def plot_four_samples(cifar_data, convert_to_int=False):
        indices = np.random.randint(0, len(cifar_data), size=4)
        fig = plt.figure()
        for i, idx in enumerate(indices):
            sample = cifar_data[idx]

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
            image = np.int32(sample["image"]) if convert_to_int else sample["image"]
            plt.imshow(image)
        plt.show()
        plt.close()

    cifar_type = 100
    # initialize data set with only a few classes and plot
    cifar100_data = CifarDataSet(root_dir="./", train=True, classes=[1], cifar_type=cifar_type,
                                 image_normalization=None)
    plot_four_samples(cifar100_data, convert_to_int=True)

    # change the current available classes and plot again
    cifar100_data.select_new_partition([5])
    plot_four_samples(cifar100_data, convert_to_int=True)

    # initialize data set  with normalization and only a few classes and plot
    cifar100_data = CifarDataSet(root_dir="./", train=True, classes=[22, 24], cifar_type=cifar_type,
                                 image_normalization="max", label_preprocessing="one-hot")
    plot_four_samples(cifar100_data, convert_to_int=False)


if __name__ == "__main__":
    main()
