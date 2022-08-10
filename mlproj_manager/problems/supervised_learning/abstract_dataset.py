from torch.utils.data import Dataset


class CustomDataSet(Dataset):
    """ General purpose abstract class for data sets used in supervised learning"""

    def __init__(self, root_dir):
        """
        :param root_dir (string): path to directory containing the data
        """
        self.root_dir = root_dir
        self.data = list()

    def load_data(self):
        """
        Loads the data set
        :return: return raw data
        """
        raise NotImplementedError("This method must be implemented for each individual data set!")

    def preprocess_data(self):
        """
        Preprocesses the data according to pre-specified parameters. Modifies self.data.
        :return: None
        """
        pass

    def __len__(self) -> int:
        """

        :return: (int) Returns the sample size of the data set.
        """
        raise NotImplementedError("This method must be implemented for each individual data set!")

    def __getitem__(self, idx) -> dict:
        """
        Returns a dictionary with two keys "image" and "label" which correspond to the given idx
        :param idx: index to retrieve data from
        :return: dictionary with sample
        """
        raise NotImplementedError("This method must be implemented for each individual data set!")
