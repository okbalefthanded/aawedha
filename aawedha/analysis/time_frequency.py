from mne.time_frequency import psd_array_multitaper
from scipy.fft import fft
from copy import deepcopy
from scipy import signal
import numpy as np


def spectral_power(data, subject=0, channel='POz'):
    """Calculate spectral power for each event in a given dataset
    for a subject at a specified channel

    Parameters
    ----------
    data : dataset instance
        epoched EEG signal dataset
    subject : int | list | str
        int : given index for a subject, by default 0 (first subject in the dataset)
        list : indices subject's subset.
        str : 'all', mean of psd across all subjects in data.
    channel : str, optional
        electrode name, by default 'Poz'

    Returns
    -------
    pwr : list of nd array (1 x samples/2 + 1)
        spectral power of each event in the dataset

    frequencies: nd array
        array of frequencies: 0 - nyquist
    """

    if isinstance(data.epochs, list) or data.epochs.ndim == 4:
        if subject == 'all':
            subjects = range(len(data.epochs))
            epochs = np.concatenate([data.epochs[idx] for idx in subjects], axis=-1)
            y = np.concatenate([data.y[idx] for idx in subjects], axis=-1)
            samples, _ ,trials = epochs.shape
            ev_count = np.unique(np.concatenate([ev for ev in data.events])).size
        else:
            samples, _, trials = data.epochs[subject].shape
            y = deepcopy(data.y[subject])
            ev_count = np.unique(data.events[subject]).size
            epochs = data.epochs[subject]
    else:
        samples, _, trials = data.epochs.shape
        y = deepcopy(data.y)
        ev_count = np.unique(data.events).size
        epochs = data.epochs

    if data.paradigm.__class__.__name__ == 'ERP':
        ev_count = 2
    else:
        y -= 1

    nyquist = data.fs / 2
    # frequencies = np.linspace(0, nyquist, np.floor(samples/2).astype(int)+1)
    # frequencies = np.arange(0, nyquist, data.fs/samples)
    frequencies = np.round(np.arange(0, nyquist, data.fs/samples), 14)

    # trials_per_class = np.round(trials / ev_count).astype(int)
    # power = np.zeros((len(frequencies), trials_per_class))
    power = []
    pwr = []
    if channel == 'all':
        ch = range(len(data.ch_names))
    else:
        ch = [data.ch_names.index(channel)]    

    for fr in range(ev_count):
        tmp = []
        for c in ch:
            epo_frq = epochs[:, c, y == fr].squeeze()
            tr_per_class = np.sum(y == fr)
            for tr in range(tr_per_class):
                f = fft(epo_frq[:, tr]) / samples
                f = 2*np.abs(f)
                f = f[0:len(frequencies)]
                # power[:, tr] = f
                power.append(f)
            # tmp.append(power.mean(axis=-1))
            tmp.append(np.mean(power, axis=0))
        # pwr.append(power.mean(axis=-1))
        pwr.append(tmp)

    if len(ch) == 1:
        pwr = [p.pop() for p in pwr]
    return pwr, frequencies


def spectral_power_welch(data, subject=0, channel='POz'):
    if isinstance(data.epochs, list) or data.epochs.ndim == 4:
        if subject == 'all':
            subjects = range(len(data.epochs))
            x = np.concatenate([data.epochs[idx] for idx in subjects], axis=-1)
            y = np.concatenate([data.y[idx] for idx in subjects], axis=-1)
            ev_count = np.unique(np.concatenate([ev for ev in data.events])).size
        else:
            y = deepcopy(data.y[subject])
            ev_count = np.unique(data.events[subject]).size
            x = data.epochs[subject]
    
    if not 0. in y:
        y -= 1

    if channel == 'all':
        ch = range(len(data.ch_names))
    else:
        ch = [data.ch_names.index(channel)]
    
    frequencies, ps = signal.welch(x, fs = data.fs, 
                                   nperseg = x.shape[0], 
                                   window = 'hann', noverlap = 0, 
                                   scaling = 'spectrum', return_onesided=True, 
                                   axis=0 )# units uV**2

    pwr = []
    for fr in range(ev_count):
        epo_frq = ps[:, ch, y == fr].squeeze()  
        pwr.append(np.mean(epo_frq, axis=-1))
    return pwr, frequencies

def spectral_power_multitaper(data, subject=0, channel='POz'):
    x = data.epochs[subject].transpose((2,1,0))
    y = deepcopy(data.y[subject]) 
    if not 0. in y:
        y -= 1
    ev_count = np.unique(data.events[subject]).size
    if channel == 'all':
        ch = range(len(data.ch_names))
    else:
        ch = [data.ch_names.index(channel)]
    ps, frequencies = psd_array_multitaper(x, data.fs, adaptive=True,
                                  normalization='full', verbose=0)
    
    pwr = []
    for fr in range(ev_count):
        epo_frq = ps[y == fr, ch, :]          
        pwr.append(np.mean(epo_frq, axis=0))
    return pwr, frequencies


def wavelet(data, subject=0, channel='POz', w=4.):
    """Calculate continuous wavelet transform for a specific subject
    in a dataset at a specific channel location, using Morelet wavelet

    Parameters
    ----------
    data : dataset instance
        epoched EEG signal dataset
    subject : int
        given index for a subject, by default 0 (first subject in the dataset)
    channel : str, optional
        electrode name, by default 'POz'

    w = float
        wavelet width

    Returns
    -------
    cwtm : list of nd array (frequencies x time)
        real continuous wavelet transform values
    t : nd array
        time vector (length of an epoch)
    frequencies : nd array
        frequencies in signal from 1 to nyquist
    """
    fs = data.fs
    ch = data.ch_names.index(channel)
    
    if isinstance(data.epochs, list) or data.epochs.ndim == 4:
        samples, channels, trials = data.epochs[subject].shape
        y = data.y[subject]
        ev_count = np.unique(data.events[subject]).size
        sig = data.epochs[subject][:, ch, :]
    else:
        samples, channels, trials = data.epochs.shape
        y = data.y
        ev_count = np.unique(data.events).size
        sig = data.epochs[:, ch, :]    
    
    # samples, channels, trials = data.epochs[subject].shape
    # y = data.y[subject]  
    
    t = np.linspace(0, samples/fs, samples)
    freq = np.linspace(1, fs/2, samples//2)
    widths = w*fs / (2*freq*np.pi)
    cwtm = []

    for ev in range(ev_count):
        tmp = []
        epo = sig[:, y == ev+1]
        for i in range(epo.shape[1]):
            tmp.append(signal.cwt(epo[:, i], signal.morlet2, widths, w=w))

        tmp = np.abs(np.array(tmp)).mean(axis=0)
        cwtm.append(tmp)

    return cwtm, t, freq


def snr(power, freqs, fs, neighbor=2):
    """Calculate frequency stimulation SNR
    
    Reference:
    "Brain–computer interface based on intermodulation frequency"
    Xiaogang Chen, Zhikai Chen, Shangkai Gao and Xiaorong Gao
    Journal of Neural Engineering, Volume 10, Number 6
    https://doi.org/10.1088/1741-2560/10/6/066009

    Parameters
    ----------
    power : list of 1d array
        spectral power
    freqs : list
        SSVEP stimuli
    fs : int
        sampling rate
    neighbor : int
        number of neighboring frequencies
    Returns
    -------
    1d array
        ssvep frequencies SNR
    """
    nyquist = fs / 2
    samples = len(power[0])
    # frequencies = np.linspace(0, nyquist, samples)
    frequencies = np.arange(0, nyquist, nyquist/samples)
    if freqs[0] == 'idle':
        power = power[1:]
        freqs = freqs[1:]
    snr_ = []
    for i, f in enumerate(freqs):
        idx = np.where(np.isclose(frequencies, float(f)))[0][0]
        adj = [idx + r for r in range(-neighbor, neighbor) if r != 0]        
        # idx = np.logical_or.reduce([frequencies == float(freqs[i])+r for r in range(-4,4) if r !=0])
        s = power[i][idx] / np.mean(power[i][adj])
        snr_.append(s.item())
    return np.array(snr_)
