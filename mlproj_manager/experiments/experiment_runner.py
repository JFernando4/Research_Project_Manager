"""
Generic running function for running experiments from command line
"""

import time
import json
import os
import argparse
from importlib.machinery import SourceFileLoader
# from project files
from mlproj_manager.experiments.abstract_experiment import Experiment
from mlproj_manager.experiments.register_experiment import load_experiment_registry, register_experiment


def parse_arguments():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--json_config_string", action="store", type=str, required=True,
                           help="json file in string format that cntains all the experiment parameters.")
    arguments.add_argument("--exp_file_path", action="store", type=str, default="", required=False,
                           help="path to the python file contain the experiment.")
    arguments.add_argument("--results_dir", action="store", type=str, required=True,
                           help="directory in which to store the results of the experiment.")
    arguments.add_argument("--exp_name", action="store", type=str,default="", required=True,
                           help="name of the experiment.")
    arguments.add_argument("--exp_class_name", action="store", type=str,default="", required=False,
                           help="name of the experiment class.")
    parsed_args = arguments.parse_args()
    return parsed_args


def main():
    # parse arguments
    parsed_args = parse_arguments()

    # get experiment name
    exp_name = parsed_args.exp_name

    # read json file from arguments
    exp_dict = json.loads(parsed_args.json_config_string)

    # create the directory that is going to store the results
    exp_dir = parsed_args.results_dir
    os.makedirs(exp_dir, exist_ok=True)

    # set up appropriate experiment
    experiment_registry = load_experiment_registry()
    in_registry = (experiment_registry is not None) and (exp_name in experiment_registry.keys())

    # retrieve experiment file path and class name
    if in_registry:
        exp_file_path, exp_class_name = experiment_registry[exp_name]
    else:
        assert hasattr(parsed_args, "exp_file_path") and hasattr(parsed_args, "exp_class_name"), \
            "Cannot register experiment without exp_file_path and exp_class_name"
        exp_file_path = parsed_args.exp_file_path
        exp_class_name = parsed_args.exp_class_name
        register_experiment(exp_name, exp_file_path, exp_class_name)

    # load experiment module
    initial_time = time.perf_counter()
    experiment_module = SourceFileLoader(exp_class_name, exp_file_path).load_module()
    exp_class = getattr(experiment_module, exp_class_name)
    final_time = time.perf_counter()
    print("Time it took to load module in minutes: {0:.2f}".format((final_time - initial_time) / 60))
    assert issubclass(exp_class, Experiment)
    
    # initialize experiment
    initial_time = time.perf_counter()
    exp = exp_class(exp_params=exp_dict,
                    results_dir=exp_dir,
                    run_index=exp_dict["index"],
                    verbose=exp_dict["verbose"])

    # run experiment
    exp.run()
    # save results
    exp.store_results()
    # print run time
    final_time = time.perf_counter()
    print("The running time in minutes is: {0:.2f}".format((final_time - initial_time) / 60))


if __name__ == "__main__":
    main()
