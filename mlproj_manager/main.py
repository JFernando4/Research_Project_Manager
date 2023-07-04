import os
import time
import argparse
import sys
from importlib.machinery import SourceFileLoader
# from project files:
from mlproj_manager.file_management import read_json_file, get_param_values, get_experiment_dir, get_missing_indices, \
    write_slurm_file, override_slurm_config
from mlproj_manager.definitions import ROOT, EXPERIMENT_RUNNER
from mlproj_manager.experiments import load_experiment_registry
sys.path.append(ROOT)


def get_experiments_dictionaries(exp_config_dict: dict):
    """
    Creates a list of dictionaries. Each dictionary represents a different parameter setting
    :param exp_config_dict: the dictionary with all the details about the experiment
    :return: list of dictionaries
    """

    # check if we're running a specific set of parameter combination and create dictionaries
    run_specific_parameter_combination = ("specific_parameter_combination" in exp_config_dict.keys())
    if run_specific_parameter_combination:
        return get_experiments_dictionaries_for_specific_parameter_combinations(exp_config_dict)

    # create dictionaries for all combinations of values of the parameters in exp_config_dict["learning_parameters"]
    init_dict = {**exp_config_dict["file_management"], **exp_config_dict["experiment_params"]}
    exp_dictionaries = [init_dict]
    for param, values in exp_config_dict["learning_params"].items():
        new_dicts = []
        for exp_dict in exp_dictionaries:
            param_values = get_param_values(values)
            for val in param_values:
                temp_dict = {**exp_dict, param: val}
                new_dicts.append(temp_dict)
        exp_dictionaries = new_dicts
    return exp_dictionaries


def get_experiments_dictionaries_for_specific_parameter_combinations(exp_config_dict: dict):
    """
    Creates a list of dictionaries for a set of specific parameter combination
    :param exp_config_dict: the dictionary with all the details about the experiment
    :return list of dictionaries, each corresponding to a specific parameter combination
    """

    exp_dicts = []
    for parameter_combination in exp_config_dict["specific_parameter_combination"]:
        assert isinstance(parameter_combination, dict)
        temp_exp_dict = {**exp_config_dict["file_management"], **exp_config_dict["experiment_params"],
                         **parameter_combination}
        exp_dicts.append(temp_exp_dict)
    return exp_dicts


def retrieve_indices(experiments_dicts: list, exp_config_dict: dict):
    """
    Retrieves the missing indices for an experiment, e.g., if an experiment requires 30 runs, but there are already
    results for 5 runs, then this function figures out that there are only 25 experiments to run and their corresponding
    indices
    :param experiments_dicts: list with the dictionaries containing the parameters for each experiment
    :param exp_config_dict: dictionary containing the experiment parameters
    :return: a list of tuples containing (experiment_dictionary, experiment_directory, missing_indices) for each
             experiment that have yet to finish all their runs
    """
    valid_experiment_dicts = []
    for exp_dict in experiments_dicts:
        exp_dir = get_experiment_dir(exp_dict, exp_dict["relevant_parameters"], exp_dict["results_path"],
                                     exp_dict["experiment_name"])
        missing_indices = get_missing_indices(exp_dir, exp_config_dict["experiment_params"]["runs"])
        if missing_indices.size > 0:
            valid_experiment_dicts.append({"dict": exp_dict, "dir": exp_dir, "indices": missing_indices})
    return valid_experiment_dicts


def run_jobs(valid_experiment_dicts: list, exp_name: str, use_slurm: bool, slurm_config=None, verbose=False):
    """
    Runs jobs in serially in the local machine or schedules the jobs using slurm
    :param valid_experiment_dicts:  (list) list of dictionaries containing the parameters for the experiments
    :param exp_name: (str) name of the experiment ot be run
    :param use_slurm: (bool) indicates whether to schedule jobs using slurm
    :param slurm_config: (str or None) config file for scheduling slurm jobs
    :param verbose:(bool) indicates whether to print status messages
    :return:
    """

    for experiment in valid_experiment_dicts:
        os.makedirs(experiment["dir"], exist_ok=True)
        missing_indices = list(experiment["indices"])
        if use_slurm:       # run slurm jobs
            assert slurm_config is not None
            run_slurm_jobs(experiment, missing_indices, slurm_config, exp_name, verbose=verbose)
        else:               # run jobs serially in machine
            run_serial_jobs(experiment, missing_indices, exp_name, verbose=verbose)


def run_slurm_jobs(experiment: dict, missing_indices: list, slurm_config: dict, exp_name: str, verbose=False):
    """
    Groups experiments according to the maximum number of rums per job in slurm_config and creates a corresponding
    bash file to schedule the job using slurm, then it schedules the job
    :param experiment: (dict) dictionary with all the parameters necessary to run the experiment
    :param missing_indices: (list) indices of the remaining experiments to be run
    :param slurm_config: (dict) dictionary with the slurm parameters
    :param exp_name: name of experiment
    :param verbose:(bool) indicates whether to print status messages
    """

    num_processed = 0           # number of processed runs
    current_job = 0             # current job number
    current_exp_batch = []      # list of experiments to be run in batch

    while num_processed != len(missing_indices):    # continue until all missing indices have been scheduled
        temp_job_num = num_processed + len(current_exp_batch)
        temp_dict = {**experiment["dict"], "index": int(missing_indices[temp_job_num]),
                     "plot_results": False, "verbose": verbose, "debug": False}
        current_exp_batch.append(temp_dict)

        # when batch size matches max_runs_per_job or when this is the last missing index, then schedule job
        if len(current_exp_batch) == slurm_config["max_runs_per_job"] or temp_job_num + 1 == len(missing_indices):
            # write slurm file
            job_path = write_slurm_file(slurm_config=slurm_config,
                                        exps_config=current_exp_batch,
                                        exp_wrapper=EXPERIMENT_RUNNER,
                                        exp_dir=experiment["dir"],
                                        job_number=current_job,
                                        exp_name=exp_name)
            # run job
            os.system("sbatch {0}".format(job_path))
            # restart everything and increment variables
            current_job += 1
            num_processed += len(current_exp_batch)
            current_exp_batch = []
            # good practice to leave some time between scheduling jobs
            time.sleep(0.5)


def run_serial_jobs(experiment: dict, missing_indices: list, exp_name: str, verbose=False):
    """
    For running job serially in the local machine
    :param experiment: (dict) dictionary with all the parameters necessary to run the experiment
    :param missing_indices: (list) indices of the remaining experiments to be run
    :param exp_name: (str) name of the experiment
    :param verbose:(bool) indicates whether to print status messages
    """
    exp_registry = load_experiment_registry()
    if exp_name not in exp_registry:
        raise KeyError("Make sure to register the experiment first!")
    exp_path, exp_class_name = exp_registry[exp_name]
    experiment_module = SourceFileLoader(exp_class_name, exp_path).load_module()
    exp_class = getattr(experiment_module, exp_class_name)

    for idx in missing_indices:
        print("Running index: {0}".format(idx))
        temp_dict = {**experiment["dict"], "index": int(idx), "plot_results": False, "debug": False}
        exp = exp_class(temp_dict, experiment["dir"], run_index=idx, verbose=verbose)
        exp.run()
        exp.store_results()


def main():

    arguments = argparse.ArgumentParser()
    arguments.add_argument("--experiment-name", action="store", type=str, default="baseline")
    arguments.add_argument("--experiment-config-path", action="store", type=str, required=True)
    arguments.add_argument("--slurm-config-path", action="store", type=str, required=False)
    arguments.add_argument("--use-slurm", action="store_true", default=False)
    arguments.add_argument("--verbose", action="store_true", default=False)
    parsed_args = arguments.parse_args()

    experiment_name = parsed_args.experiment_name

    # read experiment configuration
    exp_config_path = parsed_args.experiment_config_path
    exp_config = read_json_file(exp_config_path)

    # read slurm configuration
    slurm_config = None
    if parsed_args.use_slurm:
        slurm_config_path = parsed_args.slurm_config_path
        slurm_config = read_json_file(slurm_config_path)
        os.makedirs(slurm_config["output_dir"], exist_ok=True)
        override_slurm_config(slurm_config, exp_config)

    # create a dictionary for each experiment
    experiment_dicts = get_experiments_dictionaries(exp_config)

    # retrieve the indices for each experiment
    valid_experiment_dicts = retrieve_indices(experiment_dicts, exp_config)

    # for each valid index generate slurm file and run it
    run_jobs(valid_experiment_dicts, experiment_name, use_slurm=parsed_args.use_slurm, slurm_config=slurm_config,
             verbose=parsed_args.verbose)


if __name__ == '__main__':
    main()
