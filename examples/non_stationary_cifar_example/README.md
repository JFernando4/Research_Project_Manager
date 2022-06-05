# Experiments

An experiment is defined by the experiment's python script file, in this case the 
`non_stationary_cifar_exampl.py`, and the experiment's json config file, in this case the 
`./config_files/backprop_baseline.json` file.
The python script file simply has to create a new subclass of the abstract experiment class 
in `src/experiments/abstract_experiment.py`.
The json config file must follow this format:

    ```json
        {"level 1": {
                "level2": bleh
            }
        }

    ``

