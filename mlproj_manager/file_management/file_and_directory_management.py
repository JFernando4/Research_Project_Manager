import os
import numpy as np
import json
from torch import is_tensor
from mlproj_manager.file_management.experiment_management import get_dims_and_dtype_of_npy_file


def get_experiment_dir(exp_dictionary: dict, relevant_variables: list, result_path: str, experiment_class_name: str):
    """
    Creates a path for an experiment according to the relevant hyper-parameters of the experiment
    :param exp_dictionary: (dict) dictionary with all the experiment variables
    :param relevant_variables: (list of strings) keys used for identifying the defining variables of the experiment.
                               For example, in a supervised learning experiment, the relevant hyperparameters could
                               be the type of optimizer and the stepsize
    :param result_path: (str) path to the directory in which to store results
    :param experiment_class_name: (str) name of the experiment broader class of the experiment
    :return: path
    """

    exp_name = []
    for relevant_var in relevant_variables:
        temp_str = relevant_var + "-"
        if isinstance(exp_dictionary[relevant_var], tuple) or isinstance(exp_dictionary[relevant_var], list):
            temp_str += "-".join(str(i) for i in exp_dictionary[relevant_var])
        else:
            temp_str += str(exp_dictionary[relevant_var])
        exp_name.append(temp_str)

    exp_path = os.path.join(result_path, experiment_class_name, "_".join(exp_name))
    return exp_path


# ---*---*---*---*---*---*---*---*---  Barbed Wire ---*---*---*---*---*---*---*---*--- #
# ---*---*---*---*---*---*--- For saving and writing files ---*---*---*---*---*---*--- #
def save_experiment_results(results_dir: str, run_index: int, **kwargs):
    """
    Stores the results of an experiment. Each keyword argument correspond to a different type of result. The function
    creates a new directory for each result and store each different result in the corresponding directory in a file 
    named index-j.npy, for j = results_index
    :param results_dir: (str) path to the directory to save the results to
    :param run_index: (int) index of the run
    :param kwargs: each different keyword argument corresponds to a different result
    """
    if len(kwargs) == 0:
        print("There's nothing to save!")
        return
    save_index(results_dir, run_index=run_index)
    save_results_dict(results_dir, results_dict=kwargs, run_index=run_index)


def save_results_dict(results_dir: str, results_dict: dict, run_index=0):
    """
    Creates an npy file for each key in the dictionary. If the file already exists, it appeds to the file.
    :param results_dir: (str) path to the directory to save the results to
    :param results_dict: (dict) each key is going to be used as a directory name, use descriptive names
    :param run_index: (int) index of the run
    """
    attempts = 10
    for results_name, results in results_dict.items():
        temp_results = results if not is_tensor(results) else results.cpu().numpy()
        temp_path = os.path.join(results_dir, results_name)
        os.makedirs(temp_path, exist_ok=True)
        results_path = os.path.join(temp_path, "index-" + str(run_index) + ".npy")
        for i in range(attempts):
            try:
                np.save(results_path, temp_results)
                np.load(results_path)
                break
            except ValueError:
                print("Attempt number: {0}".format(i + 1))
                print("Something went wrong when storing results at:\n\t{0}".format(results_path))
        # np.save(os.path.join(temp_path, "index-" + str(run_index) + ".npy"), temp_results)
        print("{0} was successfully saved!".format(results_name))


def save_index(results_dir: str, run_index: int):
    """
    Stores the index of an experiment
    :param results_dir: (str) path to the directory to save the results to
    :param run_index: (dict) each key is going to be used as a directory name, use descriptive names
    """
    idx_file_path = os.path.join(results_dir, "experiment_indices.npy")
    if os.path.isfile(idx_file_path):
        index_array = np.load(idx_file_path)
        index_array = np.append(index_array, run_index)
    else:
        index_array = np.array(run_index)

    np.save(idx_file_path, index_array)
    print("Index successfully saved!")


def write_slurm_file(slurm_config: dict, exps_config: list, exp_wrapper: str, exp_dir: str, exp_name: str, job_number=0):
    """
    Creates a temporary slurm file for an experiment
    :param slurm_config: slurm parameters for running the experiment
    :param exps_config: list of experiment parameters
    :param exp_wrapper: path to a file that can run the experiment by passing a json file string to it
    :param exp_dir: directory to save all the data about the experiment
    :param exp_name: name of the experiment
    :param job_number: run number
    :return: path to the file
    """

    job_path = os.path.join(exp_dir, "job_{0}.sh".format(job_number))

    with open(job_path, mode="w") as job_file:
        job_file.writelines("#!/bin/bash\n")
        job_file.writelines("#SBATCH --job-name={0}_{1}\n".format(slurm_config["job_name"], job_number))
        output_path = os.path.join(slurm_config["output_dir"], slurm_config["output_filename"])
        job_file.writelines("#SBATCH --output={0}_{1}.out\n".format(output_path, job_number))
        job_file.writelines("#SBATCH --time={0}\n".format(slurm_config["time"]))
        job_file.writelines("#SBATCH --mem={0}\n".format(slurm_config["mem"]))
        job_file.writelines("#SBATCH --mail-type={0}\n".format(slurm_config["mail-type"]))
        job_file.writelines("#SBATCH --mail-user={0}\n".format(slurm_config["mail-user"]))
        job_file.writelines("#SBATCH --cpus-per-task={0}\n".format(slurm_config["cpus-per-task"]))
        job_file.writelines("#SBATCH --account={0}\n".format(slurm_config["account"]))
        if "gpus-per-node" in slurm_config.keys():
            job_file.writelines("#SBATCH --gpus-per-node={0}\n".format(slurm_config["gpus-per-node"]))

        job_file.writelines("export PYTHONPATH={0}\n".format(slurm_config["main_dir"]))
        job_file.writelines("source {0}/venv/bin/activate\n".format(slurm_config["main_dir"]))

        for config in exps_config:
            json_string = json.dumps(config).replace('"', '\\"')
            job_file.writelines('python3 {0} --json_config_string "{1}" --exp_name {2} --results_dir {3}\n\n'.format(
                exp_wrapper, json_string, exp_name, exp_dir))

        job_file.writelines("deactivate\n")

    return job_path


def save_experiment_config_file(results_dir: str, exp_params: dict, run_index: int):
    """
    Stores the configuration file of an experiment
    :param results_dir: (str) where to store the experiment results to
    :param exp_params: (dict) dictionary detailing all the parameters relevant for running the experiment
    :param run_index: (int) index of the run
    """
    temp_path = os.path.join(results_dir, "config_files")
    os.makedirs(temp_path, exist_ok=True)
    with open(os.path.join(temp_path, "index-" + str(run_index) + ".json"), mode="w") as json_file:
        json.dump(obj=exp_params, fp=json_file, indent=1)
    print("Config file successfully stored!")


def concatenate_results(results_dir: str, store_concatenated_results: bool = True, indices: list = None):
    """
    Given a directory containing results from different runs of an experiment where each file is named
    "index-$RUN_NUMBER.py", it reads all the files and creates a list with all the results ordered based on the
    $RUN_NUMBER in ascending order. If specified, it stores the list into a file named
    "indices-$MIN_RUN_NUMBER-$MAX_RUN_NUMBER.py"

    param results_dir: directory containing np files, each file corresponding to a different index
    param store_concatenated_results: bool indicating whether to store the list in memory
    param indices: list of int corresponding to the $RUN_NUMBER of each index
    return: np array with results
    """

    if indices is None:
        indices = get_indices(results_dir)
    concatenated_results_file_name = "indices-{0}-{1}.npy".format(indices[0], indices[-1])
    concatenated_results_file_path = os.path.join(results_dir, concatenated_results_file_name)
    if os.path.isfile(concatenated_results_file_path):
        return np.load(concatenated_results_file_path)

    results = []
    for index in indices:
        temp_file_path = os.path.join(results_dir, "index-{0}.npy".format(index))
        try:
            results.append(np.load(temp_file_path))
        except ValueError:
            print("Couldn't load file in this path: {0}".format(temp_file_path))
            raise ValueError
    results = np.array(results)

    if store_concatenated_results:
        np.save(concatenated_results_file_path, results)
        print("Results successfully saved at: {0}".format(concatenated_results_file_path))

    return results


def get_indices(results_dir: str):
    """
    Given a directory containing files named "index-$RUN_NUMBER.py", the function returns a list of indices ordered in
    ascending order

    param results_dir: path to some directory
    return: list of int corresponding to the indices of the files in ascending order
    """
    indices = []
    for file_name in os.listdir(results_dir):
        if "index" not in file_name: continue
        striped_file_name = file_name.split(".")[0]     # removes file format from file name
        run_number = striped_file_name.split("-")[1]
        indices.append(int(run_number))
    indices.sort(reverse=False)
    return indices


def get_file_paths_that_contain_keywords(dir_path: str, keyword_tuple: tuple):

    if not os.path.isdir(dir_path):
        raise ValueError("The path does not correspond to a directory.\n\tPath: {0}".format(dir_path))

    file_paths = []

    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)

        if not os.path.isfile(file_path): continue

        if all(keyword in file_name for keyword in keyword_tuple):
            file_paths.append(file_path)

    return file_paths


# ---*---*---*---*---*---*--- For loading files ---*---*---*---*---*---*--- #
def read_json_file(filepath: str):
    """
    Read a json file and returns its data as a dictionary
    :param filepath: (str) path to the file
    :return: a dictionary with the data in the json file
    """

    with open(filepath, mode="r") as json_file:
        file_data = json.load(json_file)
    return file_data


def load_experiment_results(results_dir: str, results_name: str):
    results_path = os.path.join(results_dir, results_name)
    filename_list = os.listdir(results_path)

    num_runs = len(filename_list)
    results_dims, results_dtype = get_dims_and_dtype_of_npy_file(os.path.join(results_path, filename_list[0]))

    results_array = np.zeros((num_runs, ) + results_dims, dtype=results_dtype)
    for i, filename in enumerate(filename_list):
        temp_file_path = os.path.join(results_path, filename)
        with open(temp_file_path, mode="rb") as temp_file:
            temp_results = np.load(temp_file)
        results_array[i] += temp_results
    return results_array


def get_names_for_parameter_sweep(param_combination: str, results_dir: str, return_parameter_values=False):
    """
    Given a string of the form:
        param1-val1_param2-val2_param3-*_param4-val4 (1)
    where val3 is replaced with a *, the function returns a list of names where * is replaced with each possible
    value of param3 while keeping all other param-val pairs constant.

    param param_combination: string of the same form as (1)
    param results_dir: directory with all the results with name format equal to (1) but with val3 instead of *
    param return_parameter_values: bool indicating whether to return the parameter_values
    returns: list of names as explained above
    """
    names = []
    pc_split = param_combination.split("*")

    for fn in os.listdir(results_dir):
        if pc_split[0] in fn and pc_split[1] in fn:
            param_val = fn.replace(pc_split[0], "").replace(pc_split[1], "")
            try:
                param_val = float(param_val)
            except ValueError:
                param_val = param_val

            names.append([fn, param_val])

    names.sort(key=lambda x: x[1])      # sort according to the parameter value

    if return_parameter_values:
        return [entry[0] for entry in names], [entry[1] for entry in names]
    return [entry[0] for entry in names]


# ---*---*---*---*---*---*--- For aggregating files ---*---*---*---*---*---*--- #
def bin_1d_array(array_1d: np.ndarray, bin_size: int):
    """
    Bins 1-dimensional arrays into bins of size bin_size
    :param array_1d: (np.ndarray) array to be binned
    :param bin_size: (int) size of each different bin
    :return: (np.ndarray) binned array
    """
    assert len(array_1d.shape) == 1
    return np.average(array_1d.reshape(-1, bin_size), axis=1)


def bin_results(results: np.ndarray, bin_size: int, bin_axis=1):
    """
    makes bins by adding bin_size consecutive entries in the array over the second axis of the 2D array
    :param results: a 2D numpy array
    :param bin_size: (int) number of consecutive entries to add together
    :param bin_axis: (int) axis along which to bin the results
    :return: binned 2D array of sahpe (results.shape[0], results.shape[1] // bin_size)
    """
    if bin_size == 0:
        return results
    assert results.shape[bin_axis] % bin_size == 0

    binned_results = np.apply_along_axis(lambda x: bin_1d_array(x, bin_size), axis=bin_axis, arr=results)
    return binned_results


def aggregate_results(results_dir, results_name, bin_size, bin_axis=1):
    """
    loads and bins results
    :param results_dir: (str) path to results dir
    :param results_name: (str) name of results
    :param bin_size: (int) size of bin
    :param bin_axis: (int) axis along which to bin the results
    :return: np.ndarray of binned results
    """
    results = load_experiment_results(results_dir, results_name)
    binned_results = bin_results(results, bin_size, bin_axis=bin_axis)
    return binned_results


def aggregate_large_results(results_dir, results_name, bin_size, bin_axis=0, save_results=True):
    """
    Same as aggregate_results but for larger files
    :param results_dir: (str) path to results directory
    :param results_name: (str) name of results
    :param bin_size: (int) size of bin
    :param bin_axis: (int) axis along which to bin the results
    :param save_results: (bool) whether to save the results or not
    :return:
    """
    results_path = os.path.join(results_dir, results_name)
    file_names = os.listdir(results_path)
    num_runs = len(file_names)
    dims, _ = get_dims_and_dtype_of_npy_file(os.path.join(results_path, file_names[0]))
    if bin_size == 0:
        binned_results_dims = (num_runs, dims[bin_axis])
    else:
        assert dims[bin_axis] % bin_size == 0
        binned_results_dims = (num_runs, dims[bin_axis] // bin_size)
    if len(dims) > 1: binned_results_dims += tuple(np.delete(dims, bin_axis))
    binned_results = np.zeros(binned_results_dims, dtype=np.float32)

    for i, name in enumerate(file_names):
        temp_file_path = os.path.join(results_path, file_names[i])
        temp_results = np.load(temp_file_path)
        temp_binned_results = bin_results(temp_results, bin_size, bin_axis)
        binned_results[i] += temp_binned_results

    if save_results:
        np.save(os.path.join(results_dir, results_name + "_bin-" + str(bin_size) + ".npy"), binned_results)

    return binned_results


def get_first_of_each_epoch(results_dir: str, results_name: str, epoch_length=60000, save_results=True,
                            include_last_entry=False):

    results_path = os.path.join(results_dir, results_name)
    file_names = os.listdir(results_path)
    num_runs = len(file_names)
    dims, _ = get_dims_and_dtype_of_npy_file(os.path.join(results_path, file_names[0]))
    assert len(dims) <= 2                   # must be a 2D or 1D array
    assert dims[0] % epoch_length == 0      # must be divisible by epoch length

    second_dim = dims[0] // epoch_length if not include_last_entry else dims[0] // epoch_length + 1
    new_dims = (num_runs, second_dim)
    if len(dims) > 1: new_dims += (dims[1], )

    new_results = np.zeros(new_dims, dtype=np.float32)

    for i, name in enumerate(file_names):
        temp_file_path = os.path.join(results_path, file_names[i])
        temp_results = np.load(temp_file_path)
        new_entry = temp_results[::epoch_length]
        if include_last_entry:
            new_entry = np.vstack((new_entry, temp_results[-1]))
        new_results[i] += new_entry
    if save_results:
        np.save(os.path.join(results_dir, results_name + "_first-of-epoch.npy"), new_results)

    return new_results
