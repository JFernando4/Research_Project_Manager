import os
import pickle
import re

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

        """ For creating experiment checkpoints """
        self.experiment_checkpoints_dir_path = os.path.join(self.results_dir, "experiment_checkpoints")
        self.checkpoint_identifier_name = "some_string"     # should be a string corresponding to an attribute in self
        self.checkpoint_save_frequency = 100                # should be a positive integer
        self.delete_old_checkpoints = True                  # should be a bool

    def save_experiment_checkpoint(self):
        """
        Saves all the information necessary to resume the experiment with the same index.

        The function creates a file at self.training_checkpoint_path named:
            "index-$experiment_index_${checkpoint_identifier_name}-${checkpoint_identifier_value}"

        The name should be an attribute of self defined in __init__ and the value should be an increasing sequence of
        integers where higher values correspond to latter steps of the experiment
        """

        os.makedirs(self.experiment_checkpoints_dir_path, exist_ok=True)
        checkpoint_identifier_value = getattr(self, self.checkpoint_identifier_name)

        if not isinstance(checkpoint_identifier_value, int):
            warning_message = "The checkpoint identifier should be an integer. Got {0} instead, which may result in unexpected behaviour."
            print(Warning(warning_message.format(checkpoint_identifier_value.__class__)))

        file_name = "index-{0}_{1}-{2}.p".format(self.run_index, self.checkpoint_identifier_name, checkpoint_identifier_value)
        file_path = os.path.join(self.experiment_checkpoints_dir_path, file_name)

        # retrieve model parameters and random state
        experiment_checkpoint = self.get_experiment_checkpoint()

        successfully_saved = self.create_checkpoint_file(file_path, experiment_checkpoint)

        if successfully_saved and self.delete_old_checkpoints:
            self.delete_previous_checkpoint()

    def get_experiment_checkpoint(self) -> dict:
        """ Creates a dictionary with all the necessary information to pause and resume the experiment """

        # example:
        # >> partial_results = {}
        # >> for k, v in self.results_dict.items():
        # >>    partial_results[k] = v if not isinstance(v, torch.Tensor) else v.cpu()
        #
        # >> checkpoint = {
        # >>    "model_weights": self.net.state_dict(),
        # >>    "optim_state": self.optim.state_dict(),
        # >>    "torch_rng_state": torch.get_rng_state(),
        # >>    "numpy_rng_state": np.random.get_state(),
        # >>    "cuda_rng_state": torch.cuda.get_rng_state(),
        # >>    "epoch_number": self.current_epoch,
        # >>    "current_num_classes": self.current_num_classes,
        # >>    "all_classes": self.all_classes,
        # >>    "current_running_avg_step": self.current_running_avg_step,
        # >>    "partial_results": partial_results
        # >> }
        #
        # >> return checkpoint

        raise NotImplementedError("You must implement this function if you want to checkpoint your experiment")

    def create_checkpoint_file(self, filepath: str, experiment_checkpoint: dict):
        """
        Creates a pickle file that contains the dictionary corresponding to the checkpoint

        :param filepath: path where the checkpoint is to be stored
        :param experiment_checkpoint: dictionary with data corresponding ot the current state of the experiment

        :return: bool, True if checkpoint was successfully saved
        """
        attempts = 10
        successfully_saved = False

        # attempt to save the experiment checkpoint
        for i in range(attempts):
            try:
                with open(filepath, mode="wb") as experiment_checkpoint_file:
                    pickle.dump(experiment_checkpoint, experiment_checkpoint_file)
                with open(filepath, mode="rb") as experiment_checkpoint_file:
                    pickle.load(experiment_checkpoint_file)
                successfully_saved = True
                break
            except ValueError:
                print("Something went wrong on attempt {0}.".format(i + 1))

        if successfully_saved:
            self._print("Checkpoint was successfully saved at:\n\t{0}".format(filepath))
        else:
            print("Something went wrong when attempting to save the experiment checkpoint.")

        return successfully_saved

    def delete_previous_checkpoint(self):
        """
        Deletes the previous saved checkpoint. The previous checkpoint is assumed to be stored at:
            "index-${experiment_index}_${checkpoint_identifier_name}-${previous_checkpoint_value}.p"
        where previous_checkpoint_value is:
            ${current_checkpoint_identifier_value} - ${checkpoint_save_frequency}
         """

        prev_ckpt_identifier_value = int(getattr(self, self.checkpoint_identifier_name) - self.checkpoint_save_frequency)
        file_name = "index-{0}_{1}-{2}.p".format(self.run_index, self.checkpoint_identifier_name, prev_ckpt_identifier_value)
        file_path = os.path.join(self.experiment_checkpoints_dir_path, file_name)

        if os.path.isfile(file_path):
            os.remove(file_path)
            print("The following file was deleted: {0}".format(file_path))

    def load_experiment_checkpoint(self):
        """
        Loads the latest experiment checkpoint
        """

        # find the file of the latest checkpoint
        file_name = self.get_latest_checkpoint_filename()
        if file_name == "":
            return False

        # get path to the latest checkpoint and check that it's a file
        file_path = os.path.join(self.experiment_checkpoints_dir_path, file_name)
        assert os.path.isfile(file_path)

        # load checkpoint information
        self.load_checkpoint_data_and_update_experiment_variables(file_path)
        print("Experiment checkpoint successfully loaded from:\n\t{0}".format(file_path))
        return True

    def get_latest_checkpoint_filename(self):
        """
        gets the path to the file of the last saved checkpoint of the experiment
        """
        if not os.path.isdir(self.experiment_checkpoints_dir_path):
            return ""

        latest_checkpoint_id = 0
        latest_checkpoint_file_name = ""
        for file_name in os.listdir(self.experiment_checkpoints_dir_path):
            file_name_without_extension, _ = os.path.splitext(file_name)

            # Use regular expressions to find key-value pairs
            pairs = re.findall(r'(\w+)-(\d+)', file_name_without_extension)
            index_int = int(pairs[0][1])
            ckpt_id_int = int(pairs[1][1])
            # got this from chatgpt, this is what's happening assuming filename is "index-1_ckpt_id_name-100.p"
            # >> print(file_name_without_extension)
            # >> index-1_ckpt_id_name-100
            # >> print(pairs)
            # >> [("index", "1"), ("_ckpt_id_name", "100")]

            if index_int != self.run_index:
                continue

            if ckpt_id_int > latest_checkpoint_id:
                latest_checkpoint_id = ckpt_id_int
                latest_checkpoint_file_name = file_name

        return latest_checkpoint_file_name

    def load_checkpoint_data_and_update_experiment_variables(self, file_path):
        """
        Loads the checkpoint and assigns the experiment variables the recovered values
        :param file_path: path to the experiment checkpoint
        :return: (bool) if the variables were succesfully loaded
        """

        # example:
        # >> with open(file_path, mode="rb") as experiment_checkpoint_file:
        # >>    checkpoint = pickle.load(experiment_checkpoint_file)
        #
        # >> self.net.load_state_dict(checkpoint["model_weights"])
        # >> self.optim.load_state_dict(checkpoint["optim_state"])
        # >> torch.set_rng_state(checkpoint["torch_rng_state"])
        # >> torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        # >> np.random.set_state(checkpoint["numpy_rng_state"])
        # >> self.current_epoch = checkpoint["epoch_number"]
        # >> self.current_num_classes = checkpoint["current_num_classes"]
        # >> self.all_classes = checkpoint["all_classes"]
        # >> self.current_running_avg_step = checkpoint["current_running_avg_step"]
        #
        # >> partial_results = checkpoint["partial_results"]
        # >> for k, v in self.results_dict.items():
        # >>    self.results_dict[k] = partial_results[k] if not isinstance(partial_results[k], torch.Tensor) else partial_results[k].to(self.device)

        raise NotImplementedError("You must implement this function if you want to checkpoint your experiment")

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
