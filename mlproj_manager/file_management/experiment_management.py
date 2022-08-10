import os
import numpy as np


def get_missing_indices(results_dir, sample_size):
    """
    Retrieves the indices of the completed runs of the experiment corresponding to results_dir. Then, it figures out
    what indices are missing according to the specified sample size and returns a list with those indices
    :param results_dir: experiment results directory containing a npy file named "experiment_indices.npy" with the
                        current indices
    :param sample_size: the desired number of runs for the experiment
    :return: a numpy array with the missing indices
    """

    indices_path = os.path.join(results_dir, "experiment_indices.npy")
    if os.path.isfile(indices_path):
        indices = np.arange(sample_size, dtype=np.int64)
        current_indices = np.load(indices_path)
        return np.setdiff1d(indices, current_indices)
    else:
        return np.arange(sample_size, dtype=np.int64)


def get_dims_and_dtype_of_npy_file(file_path: str):
    """
    Returns the dimensions and tata type of the array of the numpy file stored in file_path
    :param file_path: path to numpy files that contains a single numpy array
    :return: shape and data type of numpy array
    """
    temp_memmap = np.load(file_path, mmap_mode="r")
    return temp_memmap.shape, temp_memmap.dtype


def get_param_values(values):
    """
    Computes a list of parameter values based on the format of the argument values
    :param values: should be an int, str, or list. If a list of numbers, it should contain in the 0 element a
                   description of how to interpret such numbers:
    :return: a list of parameter values
    """

    if isinstance(values, int) or isinstance(values, str) or isinstance(values, float):
        return [values]

    elif isinstance(values, list):
        # if list of strings, return list
        if values[0] == "str":
            return values[1:]
        if values[0] == "tuple":
            return values[1:]

        # if list of numbers, handle according to the label in the zero entry
        elif isinstance(values[1], int) or isinstance(values[1], float):
            return create_parameter_values(values[1:], values[0])
    else:
        raise ValueError("Invalid data type: {0}".format(type(values)))


def create_parameter_values(values, param_creation_type="fixed"):
    """
    Based on a list of values, it creates a list of parameter values as specified by the param_create_type
    :param values: a list of number
    :param param_creation_type: a label indicating how to handle the list of numbers to create parameter values
                                - "fixed": return the list of numbers as is
                                - "binary": the first and second numbers correspond to the upper and lower bound,
                                            respectively and inclusive. The third number represents the number of values
                                            to be returned. New values are computed by taking the midpoint between each
                                            successive element in the list of values until reaching the desired number
                                            of values
                                -  "increment": same as binary except for the third number, which represents the
                                                increment from one value to the next one. Values are created until
                                                reaching (inclusive) or surpassing (exclusive) the upeer bound
    :return: list of values
    """

    if param_creation_type == "fixed":
        return values

    elif param_creation_type == "binary":
        assert len(values) == 3
        temp_vals = sorted(values[0:2], reverse=True)    # decreasing order
        while len(temp_vals) < values[2]:
            new_list = []
            increment = (temp_vals[0] - temp_vals[1])/2
            for val in temp_vals[1:]:
                new_list.append(val + increment)
                if len(new_list) + len(temp_vals) == values[2]:
                    break
            temp_vals = sorted(temp_vals + new_list, reverse=True)
        return temp_vals

    elif param_creation_type == "increment":
        assert len(values) == 3
        temp_vals = values[0:2]
        increment = values[2]
        current_val = values[0] + increment
        while current_val < values[1]:
            temp_vals.append(current_val)
            current_val += increment
        return temp_vals

    else:
        ValueError("Invalid case for generating new values: {0}".format(values[0]))


def override_slurm_config(slurm_dict: dict, experiment_config: dict):
    """
    Overrides slurm parameters based on the contains of experiment_config
    Use this when a specific experiment requires different resources from the one specified in the slurm json file
    :param slurm_dict: a dictionary containing the parameters for slurm
    :param experiment_config: a dictionary containing the experiment parameters
    :return: None
    """
    if "slurm_parameters" in experiment_config.keys():
        assert isinstance(experiment_config["slurm_parameters"], dict)
        for k, v in experiment_config["slurm_parameters"].items():
            slurm_dict[k] = v
