from pyLpov.utils.utils import select_target
from sklearn.metrics import accuracy_score
import numpy as np


def select_decision(scores, events, paradigm):
    """Selected spelled character in a trial

    Parameters
    ----------
    scores : 1d array
        epochs scores
    events : 1d array
        flashed characters spelled in a trial
    paradigm : Paradigm instance
        experimental dataset informations

    Returns
    -------
    str
       selected character

    Raises
    ------
    NotImplementedError
        _description_
    NotImplementedError
        _description_
    """
    if paradigm.flashing_mode == 'SC':
        command, _ = select_target(scores, events, paradigm.speller)
    elif paradigm.flashing_mode == 'RC':
        # TODO
        raise NotImplementedError
    elif paradigm.flashing_mode == 'RSP':
        # TODO
        raise NotImplementedError
    return command

    
def spelling_rate(preds, op, dataset):
    """Calculate the correct spelling rate for ERP sessions

    Parameters
    ----------
    preds : 1d or 2d array
        probabilities (neural net output) as one value or one hot vector
    op : int
        operation index, subject/fold index in dataset to test
    dataset : DataSet instance
        dataset for train/test

    Returns
    -------
    spelling rate:
        flaot : percentage of correct spelling
    """
    if hasattr(dataset, 'test_phrase'):
        phrase = dataset.test_phrase[op]
    else:
        if len(dataset.paradigm.phrase) > 1:
            phrase = dataset.paradigm.phrase[1]
        else:
            phrase = dataset.paradigm.phrase[0]
    events = dataset.test_events[op] 
    # n_char = len(phrase)
    if hasattr(dataset, 'test_flashes'):
        if dataset.test_flashes.shape[1] == 1:
            flashes = dataset.test_flashes[0]
        else:
            flashes = dataset.test_flashes[op]
        decision = decision_flexible_trials(preds, dataset, events, phrase, flashes) 
    else:
        decision = decision_fixed_trials(preds, dataset, events, phrase)
    
    return accuracy_score(phrase, decision)*100


def decision_fixed_trials(preds, dataset, events, phrase):
    n_char = len(phrase)
    trials = len(preds) // n_char
    iterations = range(0, len(preds), trials)
    decision = []
    counter = 0
    stimuli = dataset.paradigm.stimuli
    for j in iterations:
        idx = range(j, j+trials)
        if preds.shape[1] > 1:
            scores = preds[:, 1]
        else:
            scores = preds.squeeze()
        
        if len(idx) > stimuli: # single trial
            scores = scores[j:j+stimuli]
            events_per_char = events[j:j+stimuli]
        else:
            scores = scores[idx]
            events_per_char = events[idx]
        # events_per_char = events[idx]
        decision.append(select_decision(scores, events_per_char, dataset.paradigm))
        counter += 1
    
    return decision


def decision_flexible_trials(preds, dataset, events, phrase, flashes):
    n_char = len(phrase)
    trials = flashes // n_char
    decision = []
    k = 0
    stimuli = dataset.paradigm.stimuli
    for tr in np.nditer(trials):
        step = (stimuli * tr) + k
        args = np.arange(k, step)
        if preds.shape[1] > 1:
            scores = preds[:, 1]
        else:
            scores = preds.squeeze()

        if tr > stimuli: # single trial
            scores = scores[args[0]:args[0]+stimuli]
            events_per_char = events[args[0]:args[0]+stimuli]
        else:
            scores = scores[args]
            events_per_char = events[args]
        decision.append(select_decision(scores, events_per_char, dataset.paradigm))
        k = step
    return decision
