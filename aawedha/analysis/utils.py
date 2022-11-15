
import numpy as  np
import math 


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

def is_prime(n):
    """Test whether a number is prime

    Parameters
    ----------
    n : int
        a positive number

    Returns
    -------
    bool 
        True is n is prime, False otherwise.
    """
    if n % 2 == 0 and n > 2: 
        return False
    return all(n % i for i in range(3, int(math.sqrt(n)) + 1, 2))