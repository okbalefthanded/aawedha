from aawedha.analysis.time_frequency import spectral_power
from aawedha.analysis.time_frequency import wavelet
from aawedha.analysis.time_frequency import snr
from aawedha.analysis.stats import r_square_signed
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
import mne
import os


def plot_grand_average(data=None, subject=None, channel=['Cz']):
    """Plot grande average ERPs.
    Parameters
    ----------
    data : dataset instance
        epoched data
    subject : int, optional
        subject index in dataset, if None plot grand average.
    channel : list of str or 'all'
        channels to calculate average from, by default 'Cz'.
        if 'all', plot all channels in dataset       
    """

    samples = data.epochs[0].shape[0]
    time = np.linspace(0., samples/data.fs, samples) * 1000

    n_rows = 1
    n_cols = 4
    # cols = 3
    # rows = len(ch_names) // cols + 1

    if len(channel) > 1:
        if channel == 'all':
            ch = [i for i, x in enumerate(data.ch_names)]
            ch_names = data.ch_names
        else:
            ch = [i for i, x in enumerate(data.ch_names) if x in channel]
            ch_names = [data.ch_names[c] for c in ch]

        # n_rows = len(ch_names) // 4
        # n_cols = len(ch_names) // n_rows
        n_rows = len(ch_names) // n_cols + 1
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

    # if len(ch_names) % 4 != 0 and len(ch_names) > 1:
        # n_cols += 1
        # n_rows += 1
    ymin, ymax = target.min(), target.max()
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
                    col.set_ylim((ymin, ymax))
                    k += 1
        fig.tight_layout()

    plt.xlabel('[Time (ms)]')
    plt.ylabel('[µV]')
    plt.show()


def plot_spectral_power(data, subject=0, channel='POz', harmonics=2, flim=50, ylim=2,
                        save=False, savefolder=None):
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
            # ff = np.array([event, 2*event, 3*event, 4*event])
            ff = np.array([event*ev for ev in range(1, harmonics+1)])
        f_idx = np.logical_or.reduce([frequencies == f for f in ff.tolist()])

        plt.figure()
        plt.plot(frequencies, pwr[fr])
        plt.plot(frequencies[f_idx], pwr[fr][f_idx], 'ro')
        plt.xlim(0, flim)
        plt.ylim(0, ylim)
        plt.xlabel('Frequnecy [HZ]')
        plt.ylabel('Power Spectrum')
        plt.title(f'Subject: {subject + 1} Frequency: {stimuli[fr]}, at {channel}')
    if save:
        if not savefolder:
            if not os.path.exists("savedfigs"):
                os.mkdir("savedfigs")
            savefolder = "savedfigs"                
        fname = fname= f"{savefolder}/psd_{data.paradigm.filename}.png"
        plt.savefig(fname, bbox_inches='tight')


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


def plot_topomaps(data=None, channels=None, fs=512, intervals=[]):
    """Plot topomaps
    spatial distribution of data in the scalp, uses MNE plot_topomap
    Parameters
    ----------
    data : ndarray
        brain activity distributed accors channels.
    channels : list of str
        electrodes names
    fs : int
        sampling frequency of EEG recordings. by default 512 (LARESI dataset)
    """
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(channels, fs, 'eeg')
    info.set_montage(montage)
    if data.ndim == 2:
        fig, ax = plt.subplots(1, data.shape[1])
        #im = mne.viz.plot_topomap(...); plt.colorbar(im[0])
        for i in range(data.shape[1]):
            im = mne.viz.plot_topomap(data[:, i],
                                 pos=info, axes=ax[i], show=False,
                                 cmap='viridis')
            ax[i].set(title=f'[{intervals[i][0]*1000}-{intervals[i][1]*1000}] ms')
        
    else:
        im = mne.viz.plot_topomap(data, pos=info, show=False)
    plt.colorbar(im[0], fraction=0.046, pad=0.04)


def plot_snr(data=None, subject=None, channel='Oz', neighbor=2):
    """Plot frequency SNR for SSVEP based BCI paradigm

    Parameters
    ----------
    data : DataSet instance
        
    subject : int, optional
        subject index in Dataset, by default None
    channel : str, optional
        electrode name, by default 'Oz'
    """
    q = []
    
    if subject:
        pwr, _ = spectral_power(data, subject=subject, channel=channel)
        q = snr(pwr, data.paradigm.frequencies, data.fs, neighbor)
    else:
        n_subjects = len(data.epochs)
        for i in range(n_subjects):
            pwr, _ = spectral_power(data, subject=i, channel=channel)
            q.append(snr(pwr, data.paradigm.frequencies, data.fs, neighbor))
        q = np.array(q).mean(axis=0)
   
    i = np.array([float(f) for f in data.paradigm.frequencies]).argsort()
    fq = np.array(data.paradigm.frequencies)[i]
    q = q[i]

    plt.bar(fq, q)
    plt.title(f"SNR {data.title} at {channel}")
    plt.xlabel("Frequencies")
    plt.ylabel("SNR")
    plt.show()


def plot_rsigned(data=None, subject=None, channel='Cz'):
    """Plot signed r-square statistic for ERP based BCI paradigm, for a single or all subjects 
    at a single or all channels.

    Parameters
    ----------
    data : DataSet instance
        
    subject : int, optional
        subject index in dataset, by default None
        if None, plot grand average among all subjects.

    channel : str, optional
        channel name, by default 'Cz'
        if 'all', plot for each channel in data.
    """
    r = r_square_signed(data, subject)
    t = np.arange(0., len(r)/data.fs, 1/data.fs)*1000
    xlabel = '[ms]'
    ylabel = '[sgn r²]'
    legend = ['sgn r²']

    if channel != 'all':
        plt.plot(t, r[:, channel])
        plt.title(f"Sgn r² at {channel}")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(legend)
    else:
        n_rows = 1
        n_cols = 4
        n_rows = len(data.ch_names) // n_cols + 1
        ymin, ymax = r.min(), r.max()
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
        k = 0
        for row in ax:
            if isinstance(row, Iterable):
                rows = row
            else:
                rows = [row]
            for col in rows:
                if k < len(data.ch_names):
                    col.plot(t, r[:, k])
                    col.set_title(f'Sgn r² at {data.ch_names[k]}')
                    col.legend(legend)
                    col.xlabel(xlabel)
                    col.ylabel(ylabel)
                    col.set_ylim((ymin, ymax))
                    k += 1
        fig.tight_layout()

    plt.grid()
    plt.show()