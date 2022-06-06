# Experiments

An experiment is defined by the experiment's python script file, in this case the `non_stationary_cifar_exampl.py`, and the experiment's json config file, in this case the 
`./config_files/backprop_baseline.json` file. Optionally, an experiment may also use a slurm config file that specifies details about scheduling jobs using slurm. The python script file simply has to create a new subclass of the abstract experiment class 
in `src/experiments/abstract_experiment.py`. The json config file must follow this format:

    {
        "file_management": {
            "experiment_name": "distinctive_name",
            "data_path": "/path/to/data",
            "results_path": "/directory/path/where/results/are/stored",
            "relevant_parameters": ["param1","param2", "param3"]
        },
            },

        "experiment_params": {
            "epochs": 800,
            "batch_size": 1,
            "checkpoint": 1,
            "runs": 30
        },

        "learning_params": {
            "network_size": "large",
            "param1": "sgd",
            "param2": ["fixed", 0.003],
            "param3": ["fixed", 0.1]
        },

        "slurm_parameters": {
            "job_name": "dropout",
            "time": "22:00:00",
            "max_runs_per_job": 1,
        }
    }

The `"file_management"` level specifies details necessary for reading and writing operations. The results of an experiment are stored in a new directory in the path: `"results_path/experiment_name/param1-sgd_param2-0.005_param3-0.1"`. For more information about how parameters with variable value such as `"param2"` and `"param3"` are handled in the example above, check the function `create_parameter_value` located in `src/file_management/experiment_management`. 

The `"experiment_params"` level gives details about experiment variables that remain constant and are not considered part of the learning algorithm, such as number of epochs, how often to store the results, number of steps in a reinforcement learning experiments, or the batch size used to perform updates. There is no strong definition of what is considered an experiment parameter, so what may be considered an experiment parameter in one experiment may instead be considered a learning parameter on another experiment. In other words, the distinction between learning and experiment parameters is artificial, and it's just used to emphasize variables of interests (learning parameters) from variables that remain constant in an experiment (experiment parameters).

The `"learning_params"` level specifies parameters that define and influence the performance of a learning algorithm in an experiment, such as the stepsize, the dropout probability, and the weight decay factor, to name a few. 

Finally, the `"slurm_parameters"` is an optional level that's only used when using slurm for scheduling jobs. Whe using slurm, the slurm configuration is read from a different file and is used for all the experiments that share the same experiment name. However, for some algorithms it may be useful to override the details in the slurm config file. For example, some learning algorithms might take longer than others to run, in which case it might be useful to give those runs more time than the time specified in the slurm config file. Use the `"slurm_parameters"` level to do that. 


