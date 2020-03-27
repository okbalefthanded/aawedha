import matplotlib.pyplot as plt
import numpy as np
from typing import Iterable


def plot_grand_average(data=None, subject=None, channel='Cz'):
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

    if len(channel) > 1:
        ch = [i for i, x in enumerate(data.ch_names) if x in channel]
        ch_names = [data.ch_names[c] for c in ch]
    else:
        ch = data.ch_names.index(channel)
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
    target = trials[:, :, y == 1].mean(axis=-1)
    non_target = trials[:, :, y == 0].mean(axis=-1)
    n_rows = len(ch_names) // 2
    n_cols = len(ch_names) // n_rows

    if len(ch_names) % 2 != 0:
        n_cols += 1
        n_rows += 1

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
