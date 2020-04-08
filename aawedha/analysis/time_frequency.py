
from scipy.fft import fft
from scipy import signal
import numpy as np 


def spectral_power(data, subject=0, channel='Poz'):
    """Calculate spectral power for each event in a given dataset
    for a subject at a specified channel

    Parameters
    ----------
    data : dataset instance
        epoched EEG signal dataset
    subject : int
        given index for a sujbect, by default 0 (first subject in the dataset)
    channel : str, optional
        electrode name, by default 'Poz'

    Returns
    -------
    pwr : list of ndarray (1 x samples/2 + 1)
        spectral power of each event in the dataset

    frequencies: ndarray
        array of frequenices: 0 - nyquist 
    """

    samples, channels, trials = data.epochs[subject].shape
    y = data.y[subject]
    nyquist = data.fs / 2
    frequencies = np.linspace(0, nyquist, np.floor(samples/2).astype(int)+1)

    ev_count = np.unique(data.events[subject]).size

    trials_per_class = np.round(trials / ev_count).astype(int)
    pwr = []
    ch = data.ch_names.index(channel)
    power = np.zeros((len(frequencies), trials_per_class))

    for fr in range(ev_count):
        epo_frq = data.epochs[subject][:, ch, y == fr+1].squeeze()
        tr_per_class = np.sum(y == fr+1)
        for tr in range(tr_per_class):
            f = fft(epo_frq[:, tr]) / samples
            f = 2*np.abs(f)
            f = f[0:len(frequencies)]
            power[:, tr] = f
        pwr.append(power.mean(axis=-1))

    return pwr, frequencies


def wavelet(data, subject=0, channel='POz', w=4.):
    """Calculate continous wavelet transform for a specific subject
    in a dataset at a specific channel location, using Morelet wavelet

    Parameters
    ----------
    data : dataset instance
        epoched EEG signal dataset
    subject : int
        given index for a sujbect, by default 0 (first subject in the dataset)
    channel : str, optional
        electrode name, by default 'POz'

    w = float
        wavelet width

    Returns
    -------
    cwtm : list of ndarray (frequencies x time)
        real continuous wavelet transform values
    t : ndarray
        time vector (length of an epoch)
    frequencies : ndarray
        frequencies in signal from 1 to nyquist
    """
    samples, channels, trials = data.epochs[subject].shape
    y = data.y[subject]
    fs = data.fs
    ch = data.ch_names.index(channel)
    ev_count = np.unique(data.events[subject]).size

    sig = data.epochs[subject][:, ch, :]
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
