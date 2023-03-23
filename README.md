# ML Project Manager

The intent of this project is to provide a quick and easy to use framework to run machine learning experiments in a 
systematic way, while keeping track of all the important details that are necessary for reproducibility.

## Installation
This project is uploaded to the Python Package Index, so you can simply run the following command:
`python3 -m pip install mlproj_manager`

## Usage 
Here is a quick list of steps to create and run a new experiment:

1. Write a python script with a class that is a child of the `Experiment` abstract class in 
`./mlproj_manager/experiments/abstract_experiment.py`. See `./examples/non_stationary_cifar_example` for an example. 
2. Register the experiment using the command `python -m mlproj_manager.experiments.register_experiment` with the arguments
`--experiment-name` followed by a named of your choosing, `--experiment-path` followed by the path to the script
created in step 1, and `--experiment-class-name` followed by the name of the class defined in the script created in 
step 1.
3. Create a `config.json` file for your experiment that contains all the relevant details for running the experiment.
See `./examples/non_stationary_cifar_example/config_files/backprop.json` for an example.
4. Finally, run the experiment using the command `python -m mlproj_manager.main` with the arguments `--experiment-name` 
followed by the experiment name used in step 2, `--experiment-config-path` followed by the path to the config file
created in step 3, `--use-slurm` (optional) to indicate whether to schedule the experiment using slurm, and
`--slurm-config-path` (required only if using slurm) followed by the path to a similar file as the one created for step
3 but with parameters relevant to the slurm scheduler. 
