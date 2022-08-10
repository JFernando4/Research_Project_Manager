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
    for results_name, results in results_dict.items():
        temp_results = results if not is_tensor(results) else results.cpu().numpy()
        temp_path = os.path.join(results_dir, results_name)
        os.makedirs(temp_path, exist_ok=True)
        np.save(os.path.join(temp_path, "index-" + str(run_index) + ".npy"), temp_results)
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
        job_file.writelines("#SBATCH --gpus-per-node={0}\n".format(slurm_config["gpus-per-node"]))

        job_file.writelines("export PYTHONPATH={0}\n".format(slurm_config["main_dir"]))
        job_file.writelines("source {0}/venv/bin/activate\n".format(slurm_config["main_dir"]))

        for config in exps_config:
            json_string = json.dumps(config).replace('"', '\\"')
            job_file.writelines('python3 {0} --json_string "{1}" --exp_dir {2} --exp_name {3}\n\n'.format(
                exp_wrapper, json_string, exp_dir, exp_name))

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
