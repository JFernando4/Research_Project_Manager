"""
Implementation of a data loader for the data set CIFAR-100. This implementation allows the user to select specific
classes to include in or exclude from the data set.
"""
# built-in libraries
import os
# from third party packages:
import torch
import numpy as np
# from project files:
from mlproj_manager.problems.supervised_learning.abstract_dataset import CustomDataSet
from mlproj_manager.util.data_preprocessing_and_transformations import normalize, preprocess_labels
from mlproj_manager.definitions import IMAGENET_DATA_PATH


class ImageNetDataSet(CustomDataSet):
    """
    ImageNet 32x32 dataset.
    Same dataset ase the one used in the Nature paper: https://www.nature.com/articles/s41586-024-07711-7
    """

    def __init__(self,
                 root_dir=IMAGENET_DATA_PATH,
                 train=True,
                 transform=None,
                 classes: list = None,
                 image_normalization=None,
                 label_preprocessing=None,
                 use_torch=False,
                 flatten=False):
        """
        :param root_dir (string): path to directory with all the images
        :param train (booL): indicates whether to use the training or testing data set
        :param transform (callable, optional): transform to be applied to each sample
        :param classes (list): list of integers to load, integers should be in [0, 999]
        :param image_normalization (str): indicates how to normalize the data. options available:
                                            - None: no normalization
                                            - centered: subtracts the mean and divides by standard deviation
                                            - max: divides by 255, the maximum pixel value
                                            - minus-one-to-one: divides by the maximum of the data and scales to range(-1,1)
        :param label_preprocessing (str, optional): indicates how to preprocess the labels. options available:
                                                        - None: use labels as is (numbers from 0 to 9)
                                                        - one-hot: converts labels to one-hot labels
        :param use_torch (bool, optional): if true, uses torch tensors to represent the data, otherwise, uses np array
        :param flatten: whether to flatten each image
        """
        assert "classes" in os.listdir(root_dir), "No data found in the specified directory. You can download the data from here: https://drive.google.com/file/d/1i0ok3LT5_mYmFWaN7wlkpHsitUngGJ8z/view"
        super().__init__(os.path.join(root_dir, "classes"))
        assert all([i <= 999 for i in classes]), "This version of ImageNet only has 1,000 different classes."
        self.train = train
        self.transform = transform
        self.classes = classes
        self.image_norm_type = image_normalization
        self.label_preprocessing = label_preprocessing
        self.use_torch = use_torch
        self.flatten = flatten

        self.train_images_per_class = 600
        self.test_images_per_class = 100

        self.train_data, self.test_data = self.load_data()
        self.preprocess_data()
        self.current_data = self.train_data if self.train else self.test_data

    def load_data(self):
        """
        Load the train and test files.
        :return: two dictionaries with keys "data" and "labels" for the training and testing data sets
        """
        if self.classes is None:
            print("Theres nothing to load!")
        # Note: the shape of the ImageNet images is (num_samples, 3, 32, 32)
        x_train, y_train, x_test, y_test = [], [], [], []
        for idx, _class in enumerate(self.classes):
            data_file = os.path.join(self.root_dir,  f"{_class}.npy")
            new_x = np.load(data_file)
            x_train.append(new_x[:self.train_images_per_class])
            x_test.append(new_x[self.train_images_per_class:])
            y_train.append(np.array([idx] * self.train_images_per_class))
            y_test.append(np.array([idx] * self.test_images_per_class))
        x_train = torch.tensor(np.concatenate(x_train))
        y_train = torch.from_numpy(np.concatenate(y_train))
        x_test = torch.tensor(np.concatenate(x_test))
        y_test = torch.from_numpy(np.concatenate(y_test))

        train_data = {"data": x_train, "labels": y_train}
        test_data = {"data": x_test, "labels": y_test}

        return train_data, test_data


    def preprocess_data(self):
        """
        Reshapes the data into the correct dimensions, converts it to the correct type, and normalizes the data if specified
        :return: None
        """
        if self.flatten:
            self.train_data["data"] = self.train_data["data"].reshape(self.train_data["data"].shape[0], -1)
            self.test_data["data"] = self.test_data["data"].reshape(self.test_data["data"].shape[0], -1)

        if self.use_torch:
            self.train_data["data"] = torch.tensor(self.train_data["data"], dtype=torch.float32)
            self.test_data["data"] = torch.tensor(self.test_data["data"], dtype=torch.float32)

        # normalize train data
        # these three statistics below were computed using the entire dataset
        avg_px_val = 0.0003272946784095237
        stddev_px_val = 0.26002973067440316
        max_abs_px_val = 0.6303861141204834
        defaults = {"norm_type": self.image_norm_type, "avg": avg_px_val, "stddev": stddev_px_val, "max_val": max_abs_px_val}
        self.train_data["data"] = normalize(self.train_data["data"], **defaults)
        self.test_data["data"] = normalize(self.test_data["data"], **defaults)
        # preprocess labels
        self.train_data["labels"] = preprocess_labels(self.train_data["labels"], preprocessing_type=self.label_preprocessing)
        self.test_data["labels"] = preprocess_labels(self.test_data["labels"], preprocessing_type=self.label_preprocessing)

        if self.use_torch:
            self.train_data["data"] = torch.tensor(self.train_data["data"], dtype=torch.float32)
            self.train_data["labels"] = torch.tensor(self.train_data["labels"], dtype=torch.float32)
            self.test_data["data"] = torch.tensor(self.test_data["data"], dtype=torch.float32)
            self.test_data["labels"] = torch.tensor(self.test_data["labels"], dtype=torch.float32)

    def load_new_classes(self, classes: list):
        """
        Load a new set of classes into the data set
        :param classes: list of integers to load, integers should be in [0, 999]
        :return: None
        """
        self.classes = classes
        self.train_data, self.test_data = self.load_data()
        self.preprocess_data()
        self.current_data = self.train_data if self.train else self.test_data

    def set_to_training(self):
        self.train = True
        self.current_data = self.train_data

    def set_to_testing(self):
        self.train = False
        self.current_data = self.test_data

    def __len__(self):
        """
        :return (int): number of samples in the partitioned data set
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
        label = self.current_data['labels'][idx]
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
    def plot_four_samples(imagenet_data, convert_to_int=False):
        indices = np.random.randint(0, len(imagenet_data), size=4)
        fig = plt.figure()
        for i, idx in enumerate(indices):
            sample = imagenet_data[idx]

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
            image = (sample["image"] + 1) / 2
            plt.imshow(image.reshape(32, 32, 3))
        plt.show()
        plt.close()

    # initialize data set with only a few classes and plot
    imagenet = ImageNetDataSet(train=True, classes=[1, 4], image_normalization=None, label_preprocessing="one-hot")
    plot_four_samples(imagenet)

    # change the current available classes and plot again
    imagenet.load_new_classes([310, 23])
    plot_four_samples(imagenet)


if __name__ == "__main__":
    main()
