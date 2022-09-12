
import numpy as  np

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def array_to_intstr(array):
    """Convert an array of int/floats to string

    Parameters
    ----------
    array : 1d numpy array
        an array of integers in int or float format

    Returns
    -------
    1d numpy array
        array of str
    """
    return np.array([str(int(c)) for c in np.nditer(array)])
