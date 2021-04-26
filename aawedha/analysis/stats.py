import numpy as np

def r_square_signed(data=None, subject=None):
    """calculate signed r square emeasure of seperability between two classes.

    Reference:
    Benjamin Blankertz, Steven Lemm, Matthias Treder, Stefan Haufe, Klaus-Robert Müller,
    "Single-trial analysis and classification of ERP components — A tutorial",
    NeuroImage, Volume 56, Issue 2, 2011, Pages 814-825, ISSN 1053-8119,
    https://doi.org/10.1016/j.neuroimage.2010.06.048.

    Parameters
    ----------
    data : DataSet instace 
    subject :int, optional
        subject index in dataset, by default None
        if None, calculate grand average among all subjects.

    Returns
    -------
    1d array
        signed r-square between two classes in temporal axis.
    """
    if subject:
        y = data.y[subject]
        trials = data.epochs[subject]
    else:
        y = np.concatenate(data.y)
        trials = np.concatenate(data.epochs, axis=-1)
    
    target = trials[:, :, y == 1].mean(axis=-1)
    non_target = trials[:, :, y==0].mean(axis=-1)
    N1 = np.sum(y==1)
    N2 = np.sum(y==0)

    r = ((target - non_target)*np.sqrt(N1*N2)) / ((N1+N2) * trials.std(axis=-1) )
    r = r * np.abs(r)
    return r