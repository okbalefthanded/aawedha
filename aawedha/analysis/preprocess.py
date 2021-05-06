import scipy.signal as sig
import numpy as np


def bandpass(eeg, band, fs, order=2):
    """ Bandpass Filtering for Continuous/epoched EEG data

    Parameters
    ----------
    eeg: nd array (samples, channels)
        continuous/epoched EEG data

    band: list
        low_frequency, high_frequency cut_off

    fs: int
        sampling rate

    order: int
        filter order

    Returns
    -------
    nd array
        filtered EEG data
    """
    B, A = sig.butter(order, np.array(band) / (fs / 2), btype='bandpass')
    return sig.filtfilt(B, A, eeg, axis=0)


def eeg_epoch(eeg, epoch_length, markers, fs=512, baseline_correction=False, baseline=0.2):
    """Segment continuous EEG data into epochs of epoch_length following the
        stimulus onset in markers

    Parameters
    ----------
    eeg : nd array (samples, channels)
        continuous EEG data

    epoch_length : nd array (2,)
        epoch start and stop in samples

    markers : nd array (n_markers,)
        event markers onset in samples

    Returns
    -------
    eeg_epochs : nd array (samples, channels, trials)
            epoched EEG (Fortran ordering aka MATLAB format)
    """

    if baseline_correction:
        offset = baseline
    else:
        offset = 0.0
        
    start = np.around(offset*fs).astype(int)
    ep = epoch_length[0]
    epoch_length = [-start, epoch_length[1]]
    channels = int(eeg.shape[1])
    epoch_length = np.around(epoch_length)
    dur = np.arange(epoch_length[0], epoch_length[1]).reshape(
        (np.diff(epoch_length)[0], 1)) * np.ones((1, len(markers)), dtype=int)
    samples = len(dur)
    epoch_idx = dur + markers
    eeg_epochs = np.array(eeg[epoch_idx, :]).reshape(
        (samples, len(markers), channels), order='F').transpose((0, 2, 1))
    baseline = eeg_epochs.mean(axis=0)
    eeg_epochs = eeg_epochs - baseline
    eeg_epochs = eeg_epochs[start+ep:, :, :]
    return eeg_epochs
