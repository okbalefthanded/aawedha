from tensorflow.keras.utils import to_categorical
from tensorflow_addons.metrics import F1Score
import tensorflow as tf
import numpy as np


def metrics_by_lib(lib):
    """Create metrics according to the library used for evaluation.

    Parameters
    ----------
    lib : str
        Machine Learning library currently used for the evalatuion: Keras or Pytorch

    Returns
    -------
    list
        a list of metrics used for Binary classifation, in case of Keras metrics it will
        return instances, otherwise a list of metrics names.
    """
    if lib == "keras":
        return ['accuracy',
                        tf.keras.metrics.AUC(name='auc'),
                        tf.keras.metrics.TruePositives(name='tp'),
                        tf.keras.metrics.FalsePositives(name='fp'),
                        tf.keras.metrics.TrueNegatives(name='tn'),
                        tf.keras.metrics.FalseNegatives(name='fn'),
                        tf.keras.metrics.Precision(name='precision'),
                        tf.keras.metrics.Recall(name='recall'),
                        # F1Score(num_classes=2, name="f1score"),
                        ]
    else:
        return ['accuracy', 'precision', 'recall', 'auc']


def class_weights(y):
    """Calculates inverse of ratio of class' examples in train dataset
    used to re-weight loss function in case of imbalanced classes in data

    Parameters
    ----------
    y : 1d array of int
        true labels

    Returns
    -------
    cl_weights : dict of (int : float)
        class_weight : class, 1 for each class if data classes are balanced

    """
    if y.ndim == 2: # categorical_labels
        y = y.argmax(axis=1)
    if y.min() != 0:
        y = y - 1
    cl_weights = {}
    classes = np.unique(y)
    n_perclass = [np.sum(y == cl) for cl in classes]
    n_samples = np.sum(n_perclass)
    ws = np.array([np.ceil(n_samples / cl).astype(int)
                   for cl in n_perclass])
    if np.unique(ws).size == 1:
        # balanced classes
        # cl_weights = {cl: 1 for cl in classes}
        cl_weights = None
    else:
        # unbalanced classes
        if classes.size == 2:
            cl_weights = {classes[ws == ws.max()].item(
            ): ws.max(), classes[ws < ws.max()].item(): 1}
        else:
            cl_weights = {cl: ws[idx] for idx, cl in enumerate(classes)}

    return cl_weights


def labels_to_categorical(y):
    '''Convert numerical labels to categorical

    Parameters
    ----------
    y : 1d array
        true labels array in numerical format : 1,2,3,

    Returns
    -------
    y : 2d array (n_examples x n_classes)
        true labels in categorical format : example [1,0,0]
    '''
    classes = np.unique(y)
    if np.isin(0, classes):
        y = to_categorical(y)
    else:
        y = to_categorical(y - 1)
    return y


def fit_scale(X):
    """Estimate mean and standard deviation from train set
    for normalization.
    train data is normalized afterwards

    Parameters
    ----------
    X : nd array (trials, kernels, channels, samples)
        training data

    Returns
    -------
    X :  nd array (trials, kernels, channels, samples)
        normalized training data
    mu : nd array (1, kernels, channels, samples)
        mean over all trials
    sigma : nd array (1, kernels, channels, samples)
        standard deviation over all trials
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = transform_scale(X, mu, sigma)
    # X = np.subtract(X, mu[None, :, :])
    # X = np.divide(X, sigma[None, :, :])
    return X, mu, sigma


def transform_scale(X, mu, sigma):
    '''Apply normalization on validation/test data using estimated
    mean and std from fit_scale method

    Parameters
    ----------
    X :  nd array (trials, kernels, samples, channels)
        normalized training data
    mu : nd array (1, kernels, samples, channels)
        mean over all trials
    sigma : nd array (1, kernels, samples, channels)
        standard deviation over all trials

    Returns
    -------
    X :  nd array (trials, kernels, samples, channels)
        normalized data
    '''
    X = np.subtract(X, mu[None, :, :])
    X = np.divide(X, sigma[None, :, :] + 1e-7)
    return X

def transpose_split(arrays):
        """Transpose input Data to be prepared for NCHW format
        N : batch (assigned at fit), C: channels here refers to trials,
        H : height here refers to EEG channels, W : width here refers to samples

        Parameters
        ----------
        arrays: list of data arrays
            - Training data in 1st position
            - Test data in 2nd position
            - if not none, Validation data
        Returns
        ------- 
        list of arrays same order as input
        """
        for i, arr in enumerate(arrays):
            if isinstance(arr, np.ndarray):
                arrays[i] = arr.transpose((2, 1, 0))
                # trials , channels, samples
        return arrays


def aggregate_results(res):
    """Aggregate subject's results from folds into a single list

    Parameters
    ----------
    results : list of dict
            each element in the list is a dict of performance
            values in a fold

    Returns
    -------
    dict of performance metrics
    """
    results = dict()
    if type(res) is list:
        metrics = res[0].keys()
    else:
        metrics = res.keys()
    for metric in metrics:
        tmp = []
        for fold in res:
            tmp.append(fold[metric])
        results[metric] = tmp

    return results

