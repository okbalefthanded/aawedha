from pyLpov.utils.utils import select_target
from sklearn.metrics import accuracy_score


def select_decision(scores, events, paradigm):
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
    preds : n
        _description_
    op : _type_
        _description_
    dataset : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if hasattr(dataset, 'test_phrase'):
        phrase = dataset.test_phrase[op]
    else:
        if len(dataset.paradigm.phrase) > 1:
            phrase = dataset.paradigm.phrase[1]
        else:
            phrase = dataset.paradigm.phrase[0]
    events = dataset.test_events[op] 
    n_char = len(phrase)
    trials = len(preds) // n_char
    iterations = range(0, len(preds), trials)
    decision = []
    counter = 0
    for j in iterations:
        idx = range(j, j+trials)
        if preds.shape[1] > 1:
            scores = preds[idx, 1]
        else:
            scores = preds[idx]
        events_per_char = events[idx]
        decision.append(select_decision(scores, events_per_char, dataset.paradigm))
        counter += 1
    return accuracy_score(phrase, decision)*100