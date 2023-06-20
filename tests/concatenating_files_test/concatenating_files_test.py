import numpy as np
import os
import time
import sys

from mlproj_manager.file_management.concatenate_experiment_results import main as concatenate_files_main_function


def create_dummy_results(results_dir_path: str, num_runs: int, num_samples: int):
    """
    Creates fake data in results_dir_path. Creates num_runs files with format "index-$RUN_NUMBER.npy" with 100 samples
    from a brownian motion.
    """
    for i in range(num_runs):
        file_name = "index-{0}.npy".format(i)
        dummy_data = np.cumsum(np.random.normal(loc=0, scale=1, size=num_samples))  # because why not
        np.save(os.path.join(results_dir_path, file_name), dummy_data)


def main():
    np.random.seed(0)

    """ Create Fake Data """
    results_dir = "./dummy_results"

    experiment_names_and_results = {
        "exp1": ["avg", "mean", "variance"],
        "exp2": ["avg", "mean"],
        "exp3": ["variance", "avg"]}

    for experiment_name, results_names_list in experiment_names_and_results.items():
        for result_name in results_names_list:
            current_dir_path = os.path.join(results_dir, experiment_name, result_name)
            os.makedirs(current_dir_path, exist_ok=True)
            create_dummy_results(current_dir_path,
                                 num_runs=np.random.randint(10,1000),
                                 num_samples=np.random.randint(50, 1000))

    sys.argv.extend(
        ["--results-dir", results_dir,
         "--verbose",
         "--zip-original-index-files",
         "--delete-original-index-files"]
    )
    concatenate_files_main_function()


if __name__ == "__main__":
    main()
