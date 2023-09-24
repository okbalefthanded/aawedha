import numpy as np

def r_square_signed(data=None, subject=None, mode="train"):
    """calculate signed r square measure of seperability between two classes.

    Reference:
    Benjamin Blankertz, Steven Lemm, Matthias Treder, Stefan Haufe, Klaus-Robert Müller,
    "Single-trial analysis and classification of ERP components — A tutorial",
    NeuroImage, Volume 56, Issue 2, 2011, Pages 814-825, ISSN 1053-8119,
    https://doi.org/10.1016/j.neuroimage.2010.06.048.

    Parameters
    ----------
    data : DataSet instace

    subject : int, optional
        subject index in dataset, by default None
        if None, calculate grand average among all subjects.
    
    mode : str {'train', 'test'}
        Data sessins type
    Returns
    -------
    1d array
        signed r-square between two classes in temporal axis.
    """
    if mode =="train":
        epochs = data.epochs
        y = data.y
    elif mode == "test":
        if hasattr(data, "test_epochs"):
            epochs = data.test_epochs
            y = data.test_y
        else:
            raise AttributeError("DataSet instance does not has attribute test_epochs")

    if subject:
        y = y[subject]
        trials = epochs[subject]
    else:
        y = np.concatenate(y)
        trials = np.concatenate(epochs, axis=-1)
    
    target = trials[:, :, y == 1].mean(axis=-1)
    non_target = trials[:, :, y==0].mean(axis=-1)    
    
    N1 = np.sum(y==1)
    N2 = np.sum(y==0)

    r = ((target - non_target)*np.sqrt(N1*N2)) / ((N1+N2) * trials.std(axis=-1) )
    r = r * np.abs(r)

    return r

def get_erp_amplitude(target_erp, time, interval, polarity):
    """Extract max/min amplitude and latency frim target ERP data.

    Parameters
    ----------
    target_erp : 1D/2D numpy array (samples x channels) / (samples,)
        Target ERP data
    time : 1D nummpy array
        epoch times from start to end in milliseconds.
    interval : list
        interval bounds for ERP of intereset. eg [250, 350] for P300 
    polarity : str : {'pos', 'neg'}
        ERP polarity positive or negative.

    Returns
    -------
    amp : float
        ERP value in volts
    lat : lat    
        latency of ERP in milliseconds
    """
    interv_idx = np.logical_and(interval[0] <= time, time <= interval[1])
    time_interv = time[interv_idx]
    target = target_erp[interv_idx, :]
    if polarity == "pos":
        peaks = target.argmax(axis=0)
        if target.shape[1] > 1:
            amp = target.max(axis=0).mean()
        else:
             amp = target.max(axis=0).item()
    elif polarity == "neg":
            peaks = target.argmin(axis=0)
            if target.shape[1] > 1:
                amp = target.min(axis=0).mean()
            else:
                amp = target.min(axis=0).item()
    if target.shape[1] > 1:
            lat = time_interv[peaks].mean()
    else:
            lat = time_interv[peaks].item()
    return amp, lat


def erp_amplitude(data, channels, interval, polarity):
    """extract ERP component mean amplitude and latency

    Parameters
    ----------
    data : DataSet instance
        ERP dataset
    channels : list
        channels of interest where peaks are searched for.
        e.g. ['Cz', 'Pz]
    interval : list
        time window for peaks search in msec. 
        e.g. [250, 350] for P300.
    polarity : str, {'pos', 'neg'}
        polarity of ERP component, either positive or negatve.
    
    Returns
    -------
    amp: float
        mean of peak amplitudes of selected channels
    lat: float
        mean of peak amplitude's latency
    """
    ch_index = [data.ch_names.index(c) for c in channels]
    amps, lats = [], []
    online_amps, online_lats = [], []
    calib, online = {}, {}
    samples, subjects = data.epochs[0].shape[0], len(data.epochs)
    time = np.linspace(0., samples/data.fs, samples) * 1000
    for sbj in range(subjects):
        tg_idx = data.y[sbj] == 1
        target = data.epochs[sbj][:, :, tg_idx].mean(axis=-1)[:, ch_index]
        amp, lat = get_erp_amplitude(target, time, interval, polarity)
        amps.append(amp)
        lats.append(lat)
        if hasattr(data, "test_epochs"):
            tg_idx = data.test_y[sbj] == 1
            test_target = data.test_epochs[sbj][:, :, tg_idx].mean(axis=-1)[:, ch_index]
            test_amp, test_lat = get_erp_amplitude(test_target, time, interval, polarity)
            online_amps.append(test_amp)
            online_lats.append(test_lat)
    # return np.array(amps), np.array(lats)
    calib['amp'] = np.array(amps)
    calib['lat'] = np.array(lats) 
    if hasattr(data, 'test_epochs'):
         online['amp'] = np.array(online_amps)
         online['lat'] = np.array(online_lats)
    return calib, online