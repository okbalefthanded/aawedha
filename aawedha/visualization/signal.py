from aawedha.analysis.time_frequency import spectral_power, wavelet
import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable


def plot_grand_average(data=None, subject=None, channel=['Cz']):
    """Plot grande average ERPs.

    Parameters
    ----------
    data : dataset instance
        epoched data

    subject : int, optional
        subject index in dataset, if None plot grand average.

    channel : list of str
        channels to calculate average from, by default 'Cz'
    """

    samples = data.epochs[0].shape[0]
    time = np.linspace(0., samples/data.fs, samples) * 1000

    n_rows = 1
    n_cols = 1

    if len(channel) > 1:
        ch = [i for i, x in enumerate(data.ch_names) if x in channel]
        ch_names = [data.ch_names[c] for c in ch]
        n_rows = len(ch_names) // 2
        n_cols = len(ch_names) // n_rows
    else:
        ch = data.ch_names.index(channel.pop())
        ch_names = [data.ch_names[ch]]

    if subject:
        n_subjects = 1
        y = data.y[subject]
        trials = data.epochs[subject]
    else:
        n_subjects = len(data.epochs)
        y = np.concatenate(data.y)
        trials = np.concatenate(data.epochs, axis=-1)

    trials = trials[:, ch, :]
    if trials.ndim == 2:
        target = trials[:, y == 1].mean(axis=-1)
        non_target = trials[:, y == 0].mean(axis=-1)
    else:
        target = trials[:, :, y == 1].mean(axis=-1)
        non_target = trials[:, :, y == 0].mean(axis=-1)

    if len(ch_names) % 2 != 0 and len(ch_names) > 1:
        n_cols += 1
        n_rows += 1

    if len(ch_names) == 1:
        plt.plot(time, target)
        plt.plot(time, non_target)
        plt.legend(['Target', 'Non Target'])
        plt.title(f'Grand Average ERP at {ch_names.pop()}')
    else:
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
        k = 0
        for row in ax:
            if isinstance(row, Iterable):
                rows = row
            else:
                rows = [row]
            for col in rows:
                if k < len(ch_names):
                    col.plot(time, target[:, k])
                    col.plot(time, non_target[:, k])
                    col.set_title(f'Grand Average ERP at {ch_names[k]}')
                    col.legend(['Target', 'Non Target'])
                    k += 1
        fig.tight_layout()

    plt.xlabel('Time (ms)')
    plt.ylabel('µV')
    plt.show()


def plot_spectral_power(data, subject=0, channel='Poz'):
    """Plot spectral power for a given subject in a dataset at
    a specified electrode.

    Parameters
    ----------
    data : dataset instance
        epoched EEG signal dataset
    subject : int
        given index for a sujbect, by default 0 (first subject in the dataset)
    channel : str, optional
        electrode name, by default 'Poz' (suitable for SSVEP based experiments)
    """

    pwr, frequencies = spectral_power(data, subject, channel)

    if data.paradigm.title == 'ERP':
        stimuli = data.events[subject]
    else:
        stimuli = data.paradigm.frequencies

    for fr in range(len(pwr)):
        event = stimuli[fr]

        if event == 'idle':
            ff = np.zeros((3))
        else:
            event = float(event)
            ff = np.array([event, 2*event, 3*event])
        f_idx = np.logical_or(
            frequencies == ff[0], frequencies == ff[1], frequencies == ff[2])

        plt.figure()
        plt.plot(frequencies, pwr[fr])
        plt.plot(frequencies[f_idx], pwr[fr][f_idx], 'ro')
        plt.xlim(0, 50)
        plt.xlabel('Frequnecy [HZ]')
        plt.ylabel('Power Spectrum')
        plt.title(
            f'Subject: {subject + 1} Frequency: {stimuli[fr]}, at {channel}')


def plot_time_frequency(data, subject=0, channel='POz', w=4.):
    """Plot wavelet transform in a colormesh for each event in dataset

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
    """
    cwtm, t, freq = wavelet(data, subject, channel, w)

    if data.paradigm.title == 'ERP':
        stimuli = data.events[subject]
    else:
        stimuli = data.paradigm.frequencies

    for ev in range(len(cwtm)):
        fig, ax = plt.subplots()

        im = ax.pcolormesh(t, freq, cwtm[ev], cmap='viridis')
        ax.set_ylim(0, 50)
        ax.set_xlabel('Time ms')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(
            f'Subject: {subject + 1} Frequency: {stimuli[ev]}, at {channel}')
        fig.colorbar(im, ax=ax)

        plt.show()
