import yaml


def load_yaml(config_filepath):
    """Load configurations from yaml file.

    Parameters
    ----------
    config_filepath : str
        yaml configuration file path.

    Returns
    -------
    dict
        dictionary of configuration.
    """
    with open(config_filepath, 'r') as _file:
        content = _file.read()
        # load config
        config_dict = yaml.load(content)
    return config_dict