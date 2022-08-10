from abc import abstractmethod, ABC


class RLEnvironment(ABC):
    """
    Abstract class for reinforcement learning environments
    """

    def __init__(self, normalize_state: bool):
        """
        Initializes the environment
        :param normalize_state: (bool) indicates whether to normalize states before returning it to the agent
        """
        self.normalize_state = normalize_state
        self.current_state = None       # state of the environment that changes according to the agent's actions
        # add in the init any other environment constants

    @abstractmethod
    def step(self, action: int):
        """
        Modifies self.current_state based on the input action
        :param action: (int) an index corresponding to any of the available actions
        :return: new_state
        """

        raise NotImplemented("This function needs to be implemented for every RL environment.")

    @abstractmethod
    def reset(self):
        """
        Reinitializes the current state of the environment
        :return: new_state
        """

        raise NotImplemented("This function needs to be implemented for every RL environment.")

    def normalize(self, state):
        """
        Normalizes the state
        :param state: a state that is consistent with the format given to self.current_state
        :return: normalized state
        """
        return self.current_state

    def get_current_state(self):
        if self.normalize_state:
            return self.normalize(self.current_state)
        return self.current_state
