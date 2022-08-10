import numpy as np


def compute_average_and_standard_error(results:np.ndarray, axis=0):
    """
    Computes average and standard error of an array along a specified axis
    :param results: (np.ndarray) array containing the data to be aggregated
    :param axis: (int) axis along which to compute the average
    :return: (avg, standard error), same shape as the input array but with data aggregated along a specified dimension
    """
    sample_size = results.shape[axis]
    avg = np.average(results, axis=axis)
    standard_dev = np.std(results, ddof=1, axis=axis)
    standard_error = standard_dev / np.sqrt(sample_size)
    return avg, standard_error
