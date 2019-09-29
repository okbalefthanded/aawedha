import scipy.signal as sig
import numpy as np


def bandpass(eeg, band, fs, order=2):
  """
  """
  B,A = sig.butter(order, np.array(band)/(fs/2), btype='bandpass')
  return sig.filtfilt(B, A, eeg, axis=0)


def fit_normalize():
  """
  """
  NotImplementedError
