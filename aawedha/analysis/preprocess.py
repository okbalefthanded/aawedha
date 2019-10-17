import scipy.signal as sig
import numpy as np


def bandpass(eeg, band, fs, order=2):
  B,A = sig.butter(order, np.array(band)/(fs/2), btype='bandpass')
  return sig.filtfilt(B, A, eeg, axis=0)


def eeg_epoch(eeg, epoch_length, markers):
    channels = int(eeg.shape[1])
    epoch_length = np.around(epoch_length)
    dur = np.arange(epoch_length[0], epoch_length[1]).reshape((np.diff(epoch_length)[0],1)) * np.ones( (1, len(markers)),dtype=int)
    samples = len(dur)
    epoch_idx = dur + markers
    eeg_epochs = np.array(eeg[epoch_idx,:]).reshape((samples, len(markers), channels), order='F').transpose((0,2,1))
    return eeg_epochs