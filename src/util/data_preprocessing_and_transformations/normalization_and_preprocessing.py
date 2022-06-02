"""
Helper functions for normalizing data
"""
# from 3rd party packages:
import numpy as np
import torch

def normalize(data, norm_type: str, **kwargs):
    """
    Normalizes the given data according to the specified type
    :param data: (np.array or torch.Tensor) array containing the data
    :param norm_type: (str) type of normalization from the following options:
                                - None: no normalization
                                - centered: subtracts the mean and divides by standard deviation
                                - max: divides the maximum of the data
                                - minus-one-to-one: divides by the maximum of the data and scales to range(-1,1)
    :param kwargs: extra optional keyword arguments for the different normalization types. Available options:
                        - None: no available keyword arguments
                        - centered: avg = average of the data, stddev = standard deviation of the data
                        - max: max_val = maximum value of the data
                        - minus-one-to-one: max_val as above
                   any necessary statistic used for normalization is computed from the data if not give.
    :return: normalized numpy or torch array
    """
    if norm_type is None:
        return data
    elif norm_type == "centered":
        avg = np.average(np.float32(data)) if "avg" not in kwargs.keys() else kwargs["avg"]
        stddev = np.std(np.float32(data), ddof=1) if "stddev" not in kwargs.keys() else kwargs["stddev"]
        return (data - avg) / stddev
    elif norm_type == "max" or norm_type == "minus-one-to-one":
        max_val = max(data) if "max_val" not in kwargs.keys() else kwargs["max_val"]
        if norm_type == "minus-one-to-one":
            return 2 * (data / max_val) - 1
        return data / max_val
    else:
        raise ValueError("Invalid normalization type: {0}".format(norm_type))


def preprocess_labels(labels, preprocessing_type: str):
    """
    Preprocesses the given labels according to preprocessing_type
    :param labels: (numpy or torch array) array containing the labels to be preprocessed
    :param preprocessing_type: (str) type of preprocessing from the following:
                                        - None: does nothing
                                        - one-hot: turns labels into one-hot labels
    :return: preprocessed labels
    """
    if preprocessing_type is None:
        return labels
    elif preprocessing_type == "one-hot":
        return from_integers_to_one_hot(labels)
    else:
        raise ValueError("Invalid label preprocessing: {0}".format(preprocessing_type))


def from_integers_to_one_hot(labels):
    """
    Given an array of integer labels returns one-hot labels
    :param labels: (numpy or torch array) array containing integer labels
    :return: array containing one-hot labels
    """
    if isinstance(labels, np.ndarray):
        num_samples = labels.size
    elif isinstance(labels, torch.Tensor):
        num_samples = np.int64(labels.size()[0])
    elif isinstance(labels, list):
        num_samples = len(labels)
        labels = np.array(labels)
    else:
        raise ValueError("The data type of the argument labels was {0}, "
                         "but expected, np.ndarray, torch.Tensor, or list".format(type(labels)))

    max_label = np.int64(labels.max() + 1)
    one_hot_labels = np.zeros((num_samples, max_label), dtype=np.float32)
    one_hot_labels[np.arange(num_samples), np.int64(labels)] = 1
    return one_hot_labels


def from_one_hot_to_integer(labels):
    """
    Given an array of one-hot labels returns an array of integer labels
    :param labels: (numpy or torch array) array containing one-hot labels
    :return: array containing integer labels
    """
    assert len(labels.shape) == 2

    integer_labels = np.argmax(np.int64(labels), axis=1)
    return integer_labels
