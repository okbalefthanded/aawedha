from sklearn.metrics import roc_curve, confusion_matrix
from aawedha.models.utils_models import model_lib
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.metrics import F1Score
from pyLpov.utils.utils import select_target
from sklearn.metrics import accuracy_score
from pyLpov.utils.utils import itr
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

def create_split(X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """gather data arrays in a dict

    Parameters
    ----------
    X_train : ndarray (N_training, channels, samples)
        training data
    X_val : ndarray (N_validation, channels, samples)
        validation data
    X_test : ndarray (N_test, channels, samples)
        test data
    Y_train : 1d array
        training data labels
    Y_val : 1d array
        validation data labels
    Y_test : 1d array 
        test data labels

    Returns
    -------
    dict
        evaluation data split dictionary where the key is the array's name.
    """
    split = {}
    split['X_test'] = None
    split['X_val'] = None
    split['X_train'] = X_train if X_train.dtype is np.float32 else X_train.astype(np.float32)
    split['Y_train'] = Y_train
    if isinstance(X_test, np.ndarray):
        split['X_test'] = X_test if X_test.dtype is np.float32 else X_test.astype(np.float32)
    split['Y_test'] = Y_test
    if isinstance(X_val, np.ndarray):
        split['X_val'] = X_val if X_val.dtype is np.float32 else X_val.astype(np.float32)
    split['Y_val'] = Y_val
    return split

def measure_performance(Y_test, probs, perf, metrics_names):
    """Measure model performance on a dataset

    Calculates model performance on metrics and sets Confusion Matrix for
    each fold

    Parameters
    ----------
    Y_test : 2d array (n_examples x n_classes)
        true class labels in Tensorflow format

    probs : 2d array (n_examples x n_classes)
        model output predictions as probability of belonging to a class

    Returns
    -------
        dict of performance metrics : {metric : value}
    """
    results  = {}      
    # classes = Y_test.max()
    if Y_test.ndim == 2:
        Y_test = Y_test.argmax(axis=1)
        # probs = probs[:, 1]

    classes = np.unique(Y_test).size

    if isinstance(perf, dict):
        results = {metric:value for metric, value in perf.items()}
    elif isinstance(perf, list):
        results = {metric:value for metric, value in zip(metrics_names, perf)}

    if classes == 2:
        if probs.shape[1] > 1:            
            probs = probs[:, 1]
        fp_rate, tp_rate, thresholds = roc_curve(Y_test, probs)
        viz = {'fp_threshold': fp_rate, 'tp_threshold': tp_rate}
        results['viz'] = viz
        preds = np.zeros(len(probs))
        preds[probs.squeeze() > .5] = 1.
    else:
        preds = probs.argmax(axis=-1)

    return results, confusion_matrix(Y_test, preds)

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

def predict_trial(epochs, y, desc, model, n_char, commands):
    """Calculate character recognition rate
    for a single subject session.

    Parameters
    ----------
    epochs : ndarray (trials x channels x samples)
        epoched EEG session
    y : 1d array (trials) or 2d array (categorical labels) (trials x 2)
        labels 1/0 :  1 target, 0 non target
    desc : 1d array (trials)
        events in order of flashing
    model : trained Keras/Pytorch model
        trained model for prediction.
    n_char : int
        count of characters spelled in session
    commands : int
        speller characters

    Returns
    -------
    int
        percentage of characters correctly detected.
    """
    k = 0
    n_epochs = epochs.shape[0]

    trials = n_epochs // n_char
    iterations = range(0, n_epochs, trials) 
    
    labels = []
    cmds = []
    model_type = model_lib(type(model))
    for j in iterations:
        idx = range(j, j+trials)
        one_trial = epochs[idx, :, :]
        if model_type == "keras":
            scores = model.predict(one_trial)
        else:
            norm = True
            if not isinstance(model.mu, np.ndarray):
                norm = False
            scores = model.predict(one_trial, normalize=norm)
        one_desc = desc[idx]
        if y.ndim == 1:
            target_idx = np.where(y[idx] == 1)
        else:
            target_idx = np.where(y[idx, 1] == 1)
            scores = scores[:, 1]
        labels.append(one_desc[target_idx[0][0]])
        command, _ = select_target(scores, one_desc, commands)
        if command == '#':
            command = 0
        cmds.append(int(command))
        k += 1
    
    return accuracy_score(labels, cmds)*100

def char_rate_epoch(epochs, desc, model, phrase, n_char, paradigm, flashes=None):
    """_summary_

    Parameters
    ----------
    epochs : ndarray (trials x channels x samples)
        epoched EEG session
    desc : 1d array (trials)
        events in order of flashing
    model : trained Keras/Pytorch model
        trained model for prediction.
    phrase : 1d array of int 
        correct phrase to spell by subject
    n_char : int
        count of characters spelled in session
    paradigm : paradigm instance
        experiment information
    flashes : 1darray of int, optional
        number of total flashes for the spelling of each character, by default None
        if None: a fixed number of flashes across all dataset.
    Returns
    -------
    acc_per_rep: 1d array of float (trials_repetition)
        percentage of correct character recognition per trial repetition.
    """
    if isinstance(flashes, np.ndarray):
        trials = flashes // n_char # repetition per char
        scores, desc, max_step = predict_flexible_trials(epochs, desc, model, paradigm, trials)
        repetitions = np.min(trials)         
    else:
        scores = predict_fixed_trials(epochs, model)
        repetitions = paradigm.repetition
        max_step = repetitions * paradigm.stimuli 

    acc_per_rep = np.zeros((repetitions))
    commands = paradigm.speller

    for rep in range(1, repetitions+1):
        cmds = []
        for i in range(0, scores.size, max_step):
            args = np.arange(i, i+(rep*n_char))
            command, _ = select_target(scores[args], desc[args], commands)
            if command == '#':
                command = 0
            cmds.append(int(command))        
        acc_per_rep[rep-1] = accuracy_score(phrase, cmds) * 100
    
    return acc_per_rep

def predict_fixed_trials(epochs, model):
    """Calculate prediction scores on epoches with fixed number
    of trial repetitions across all dataset.

    Parameters
    ----------
    epochs : ndarray (trials x channels x samples)
        epoched EEG session
    model : trained Keras/Pytorch model
        trained model for prediction.

    Returns
    -------
    scores : 1d array (trials)
        probability score for class target 1.
    """
    model_type = model_lib(type(model))
    if model_type == "keras":
        scores = model.predict(epochs)
    else:
        norm = True
        if not isinstance(model.mu, np.ndarray):
            norm = False
        scores = model.predict(epochs, normalize=norm)
        if scores.ndim > 1:
            scores = scores[:, 1]
    return scores

def predict_flexible_trials(epochs, desc, model, paradigm, trials):    
    """Calculate prediction scores on epoches with flexible number
    of trial repetitions for each subject's session.
    a fixed number of trials is selected after prediction, this number
    equals the minimum of trial repitition in the session.

    Parameters
    ----------
    epochs : ndarray (trials x channels x samples)
        epoched EEG session
    desc : 1d array (trials)
        events in order of flashing
    model : trained Keras/Pytorch model
        trained model for prediction.
    paradigm : paradigm instance
        experiment information
    trials : 1darray
        number of trial repetition for the spelling of each character.

    Returns
    -------
    scores : 1d array (fixed_trials)
        probability score for class target 1.
    
    desc_flat : 1darray
        selected events
    
    max_step : int
        fixed total number of trial repetition for spelling of
        a single character.    
    """
    step = 0
    k = 0
    scores = []
    desc_flat = []
    repetitions = np.min(trials)
    max_step = repetitions * paradigm.stimuli 
    model_type = model_lib(type(model))
    for tr in np.nditer(trials):    
        step = (paradigm.stimuli * tr) + k
        args = np.arange(k, step)
        if model_type == "keras":
            sc = model.predict(epochs[args, :, :])[:max_step]
        else:
            norm = True
            if not isinstance(model.mu, np.ndarray):
                norm = False
            sc = model.predict(epochs[args, :, :], normalize=norm)[:max_step]        
        scores.append(sc)        
        desc_flat.append(desc.squeeze()[args][:max_step])
        k = step

    scores = np.concatenate(scores).squeeze()
    desc_flat = np.concatenate(desc_flat).squeeze()
    return scores, desc_flat, max_step
