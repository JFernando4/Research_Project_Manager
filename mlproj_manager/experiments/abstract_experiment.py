# from project files:
from mlproj_manager.file_management.file_and_directory_management import save_experiment_config_file, save_experiment_results


class Experiment:

    """ Abstract class for experiments that outlines all the methods and experiment is expected to have """
    def __init__(self, exp_params: dict, results_dir: str, run_index: int, verbose: bool = True,
                 plot_results: bool = False):
        """
        Initialize the experiment
        :param exp_params: (dict) all the information necessary to run the experiment
        :param results_dir: (str) path in which to store results to
        :param run_index: (int) index of the experiment run
        """
        self.exp_params = exp_params
        self.results_dir = results_dir
        self.run_index = run_index
        self.verbose = verbose
        self.plot_results = plot_results
        save_experiment_config_file(results_dir, exp_params, run_index)

        self.results_dict = {}

    def store_results(self):
        """
        Stores the results in self.results_dict. User should make sure that self.results_dict contains all the
        data to be stored.
        """
        save_experiment_results(self.results_dir, self.run_index, **self.results_dict)

    def _print(self, formatted_string):
        """
        print function used for debugging or displaying the progress of the experimet
        :param formatted_string: (str) text to print
        """
        if self.verbose:
            print(formatted_string)

    def run(self):
        """
        Runs the experiment
        """
        raise NotImplementedError("This function should be implemented for each different experiment!")
