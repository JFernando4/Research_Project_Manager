# Running experiments in slurm

The intention of this example is to show how to schedule experiments with slurm. 
For this project, all the tests were performed in the Beluga server of the Digital Research Alliance of Canada, 
formerly known as Compute Canada.

The python script `mnist_example.py` is simply a classification task using MNIST. 
The parameters of the experiment are specified in `config.json`. 
Given the different parameters, it would take a long time to run it serially in a local computer.
Thus, this is the kind of experiment that would be best run in parallel using slurm. 

## Config files
The experiment slurm config file contains this information:

    {
      "job_name": "mnist",
      "output_dir": "/scratch/jfernan/example/outputs",
      "output_filename": "mnist-%n-%j.out",
      "time": "4:00:00",
      "mem": "8G",
      "mail-type": "BEGIN,END,FAIL",
      "mail-user":"jfhernan@ualberta.ca",
      "cpus-per-task": 4,
      "max_runs_per_job": 10,
      "main_dir": "/home/jfernan/Research_Project_Manager",
      "account": "def-sutton",
      "gpus-per-node": "1"
    }

Let's go through all the options used in the config file:
* `job_name`: this is the name of the job. You can see the progress of the job by using this name as reference. 
Nevertheless, the name doesn't have to be unique since slurm creates a unique ID for each new job. 
* `output_dir`: path to directory where slurm can dump output files. These output files will contain anything printed 
by the script or any error messages. The symbols `%n` and `%j` are replaced by the node and job IDs, respectively. 
Thus, the name `mnist-%n-%j.out` will result in a separate output file for each different job scheduled in slurm.
* `time`: how long to run the experiment for. Make sure to give it enough time or to handle timeouts in such a way that
you don't lose any data.
* `mem`: amount of memory to use. 
* `mail-type`: types of mail to receive from slurm regarding the job status.
* `mail-user`: who to send emails to.
* `cpus-per-task`: number of cpus to use for each slurm job.
* `max_runs_per_job`: maximum number of runs to try to run in a single slurm job. All of these runs should be able to
finish running before hitting the maximum amount of time allowed.
* `main_dir`: path to the main directory of the experiment. This not necessarily the same as the path to where the 
experiment file is located, but the path to where the source code of the experiment and the `venv` are located.
* `account`: whose slurm account to use.
* `gpus-per-node`: how many gpus to use for each different node.

These are all the slurm options supported by `mlproj_manager=0.0.2`.

## How to set up and run the experiment
First, while in the `Research_Project_Manager` directory run these two lines:

    export PYTHONPATH=.
    python3 ./mlproj_manager/experiments/register_experiment.py \
        --experiment-name mnist_example --experiment-path ./examples/mnist_example/mnist_example.py \
        --experiment-class-name MNISTExperiment

This will register the experiment so that `mlproj_manager` can run it by using an experiment wrapper.
Next, schedule your jobs using the following command:

    python3 ./mlproj_manager/main.py \
        --experiment-name mnist_example --experiment-config-path ./examples/mnist_example/config.json \
        --use-slurm --slurm-config-path ./examples/mnist_example/slurm_config.json --verbose

You should see the output of running the command `sbatch` in slurm: `Submitted batch job $JOB-ID`. You'll see 
several of these depending on the number of jobs that were scheduled.

