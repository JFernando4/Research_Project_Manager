"""
Command line script for registering new experiments
"""
import pickle
import argparse
import os
from importlib.machinery import SourceFileLoader
# from project files
from mlproj_manager.definitions import ROOT
from mlproj_manager.experiments.abstract_experiment import Experiment


def load_experiment_registry():
    registry_path = os.path.join(ROOT, "experiment_registry.p")
    if os.path.isfile(registry_path):
        with open(registry_path, mode="rb") as registry_file:
            registry = pickle.load(registry_file)
        return registry
    else:
        return None


def register_experiment(experiment_name: str, experiment_path: str, experiment_class_name: str):
    """
    Creates a pickle file with a dictionary where the keys are the experiment names and the values are the paths
    :param experiment_name: (str) a name used to identify the experiment
    :param experiment_path: (str) path to the the experiment
    :param experiment_class_name: (str) name of the experiment class, not necessarily the same as experiment_name
    :return:
    """

    # check that the path is correct
    assert os.path.isfile(experiment_path)

    # load experiment module
    experiment_module = SourceFileLoader(experiment_class_name, experiment_path).load_module()

    # check that experiment is a subclass of the abstract Experiment class
    assert issubclass(getattr(experiment_module, experiment_class_name), Experiment)

    experiment_registry_path = os.path.join(ROOT, "experiment_registry.p")

    # tyr to load experiment registry, create it if it doesn't exist yet
    experiment_registry = load_experiment_registry()
    if experiment_registry is None:
        experiment_registry = {}

    # store new experiment name, path, and class name in registry
    experiment_registry[experiment_name] = (experiment_path, experiment_class_name)
    with open(experiment_registry_path, mode="wb") as registry_file:
        pickle.dump(experiment_registry, registry_file)


def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument("--experiment-name", action="store", type=str, required=True)
    arguments.add_argument("--experiment-class-name", action="store", type=str, required=False)
    arguments.add_argument("--experiment-path", action="store", type=str, required=True)
    parsed_args = arguments.parse_args()

    exp_name = parsed_args.experiment_name
    exp_path = parsed_args.experiment_path
    exp_class_name = exp_name if not hasattr(parsed_args,"experiment_class_name") else parsed_args.experiment_class_name

    register_experiment(experiment_name=exp_name, experiment_path=exp_path, experiment_class_name=exp_class_name)


if __name__ == '__main__':
    main()
