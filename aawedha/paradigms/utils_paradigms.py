from aawedha.evaluation.evaluation_utils import positive_class_prob
from pyLpov.utils.utils import select_target
from pyLpov.utils.utils import itr
from sklearn.metrics import accuracy_score
import numpy as np

def phrase_from_dataset(dataset, op):
    if hasattr(dataset, 'test_phrase'):
        phrase = dataset.test_phrase[op]
    else:
        if len(dataset.paradigm.phrase) > 1:
            phrase = dataset.paradigm.phrase[1]
        else:
            phrase = dataset.paradigm.phrase[0]
    is_uniform = True
    # TODO: check for dataset with no test epochs
    if hasattr(dataset, "test_epochs"):
        lengths = [epoch.shape[-1] for epoch in dataset.test_epochs]
        is_uniform = np.unique(lengths).size == 1    
    return phrase, is_uniform

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

    """
    if paradigm.flashing_mode.lower() == 'sc':
        command, _ = select_target(scores, events, paradigm.speller)
    elif paradigm.flashing_mode.lower() == 'rc':
        # TODO
        raise NotImplementedError
    elif paradigm.flashing_mode.lower() == 'rsp':
        # TODO
        raise NotImplementedError
    elif paradigm.flashing_mode.lower() == "mvep_bidir":
        # TODO
        raise NotImplementedError
    else:
        raise ValueError("Unknown Flashing Mode")
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
        flaot / list : percentage of correct spelling
    """
    phrase, is_uniform = phrase_from_dataset(dataset, op)
    events = dataset.test_events[op] 
    
    if hasattr(dataset, 'test_flashes') and not is_uniform:
        if dataset.test_flashes.shape[1] == 1:
            flashes = dataset.test_flashes[0]
        else:
            flashes = dataset.test_flashes[op]
        decision = decision_flexible_trials(preds, dataset, events, phrase, flashes) 
    else:
        decision = decision_fixed_trials(preds, dataset, events, phrase)

    if len(decision) == 1:
        # singla trial ERP dataset
        return accuracy_score(phrase, decision.pop())*100
    else:
        # multiple trials ERP dataset
        return np.array([accuracy_score(phrase, d)*100 for d in decision])

def decision_fixed_trials(preds, dataset, events, phrase):
    sequence = dataset.paradigm.get_repetitions()
    stimuli  = dataset.paradigm.get_stimuli()  
    n_char   = len(phrase)
    trials   = len(preds) // n_char
    iterations = range(0, len(preds), trials)
    decision = []
    counter  = 0
    scores = positive_class_prob(preds)
    for seq in range(1, sequence + 1):
        seq_decision = []
        for j in iterations:
            # idx = range(j, j+trials)        
            # if len(idx) > stimuli: # single trial
            #     p = scores[j:j+stimuli]
            #     events_per_char = events[j:j+stimuli]
            # else:
            p = scores[j:j+(seq*stimuli)]
            events_per_char = events[j:j+(seq*stimuli)]
            # print(p.shape, events_per_char.shape)
            seq_decision.append(select_decision(p, events_per_char, dataset.paradigm))
        decision.append(seq_decision)
        counter += 1
    
    return decision


def decision_flexible_trials(preds, dataset, events, phrase, flashes):
    n_char = len(phrase)
    trials = flashes // n_char
    decision = []
    k = 0
    stimuli = dataset.paradigm.stimuli
    scores = positive_class_prob(preds)
    for tr in np.nditer(trials):
        step = (stimuli * tr) + k
        args = np.arange(k, step)
        if tr > stimuli: # single trial
            p = scores[args[0]:args[0]+stimuli]
            events_per_char = events[args[0]:args[0]+stimuli]
        else:
            p = scores[args]
            events_per_char = events[args]
        decision.append(select_decision(p, events_per_char, dataset.paradigm))
        k = step
    return decision


def itr_score(score, op, dataset):
    """Calculate the ITR for All paradigms

    Parameters
    ----------
    score : 1d or 2d array
        spelling rate if paradigm is ERP, accuracy rate otherwise.
    op : int
        operation index, subject/fold index in dataset to test
    dataset : DataSet instance
        dataset for train/test

    Returns
    -------
    spelling rate:
        flaot : percentage of correct spelling
    """
    dur      = dataset.paradigm.one_trial_duration()    
    sequence = dataset.paradigm.get_repetitions()
    n = dataset.paradigm.stimuli
    t = dur * sequence
    p = score / 100 if np.any(score > 1)  else score 
    if isinstance(p, np.ndarray):        
        t = [dur*seq for seq in range(1, sequence + 1)]
        return np.array([itr(n, pi, ti) for pi, ti in zip(p, t)])
    else:
        return itr(n, p, t)

paradigm_metrics = {
    'spelling_rate': spelling_rate,
    'itr': itr_score,
}