import torch


def get_random_seeds():
    """
    This function should always return the same random seeds if the user is using the same version of pytorch as the
    project.

    The intended use for this random seeds is for an experiment to use the random seed in the array corresponding to the
    experiment index.
     """
    torch.random.manual_seed(42)
    max_num_runs = 100000             # I don't think I'll ever run an experiment with this many runs
    random_seeds = torch.randperm(int(1e5), dtype=torch.int32)[:max_num_runs]
    return random_seeds


def access_dict(dict_to_check: dict, key: str, default, val_type=None, choices=None):
    """
    Checks if a given key is in dictionary and checks the value stored at the given key is the correct type and has the
    correct value, then returns the value.
    If the given key is  not in the dictionary, it adds a default value to the dictionary and returns the default 
    value.
    :param dict_to_check: (dict)  
    :param key: (str)
    :param default: (dynamic type) value to set the key to if key_to_check is not in dict_to_check
    :param val_type: (type or None) if not None, checks if value stored at key is of the correct type
    :param choices: (list or None) if not None, checks that the value stored at key is one of the listed choices
    :return: default_value or dict_to_check[key_to_check]
    """

    if key in dict_to_check.keys():

        # check if correct type and correct
        if val_type is not None:
            assert isinstance(dict_to_check[key], val_type)
        if choices is not None:
            assert dict_to_check[key] in choices

        return dict_to_check[key]
    else:
        dict_to_check[key] = default
        return default
