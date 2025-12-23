# TODO: REFACTOR, delete TF code, use pytorch only
from aawedha.io.io_utils import load_yaml
  

def config_to_description(cfg_path):
    """convert yaml configuration entries to a list of optimizer
    description for evaluation.

    Parameters
    ----------
    cfg_path : str
        yaml configuration file path.

    Returns
    -------
    list
        list of description with list[0] is the optimizer name
        and list[1] is a dict of optimizer params.
    """
    cfg = load_yaml(cfg_path)
    # return [cfg["name"], cfg["params"]]
    return {cfg["name"]: cfg["params"]}