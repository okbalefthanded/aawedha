import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_temporal_filters(filters, fs):
    '''Temporal filters
    '''
    kernel_length = filters.shape[2]
    time = np.arange(0, kernel_length/fs, 1/fs) * 1000
    n_filters = filters.shape[-1]
    n_rows = n_filters // 2
    n_cols = n_rows // 2
    if n_filters % 2 != 0:
        n_cols += 1
        n_rows += 1

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
    k = 0
    for row in ax:
        for col in row:
            if k < n_filters:
                col.plot(time, filters[:, :, :, :, k].squeeze())
                col.set_title(f'Temporal Filter {k+1}')
                k += 1

    fig.tight_layout()
    plt.xlabel(' Time (ms) ')
    plt.show()


def plot_topomaps(data=None, channels=None, fs=512):
    '''Topographic maps
    '''
    montage = mne.channels.make_standard_montage('standard_1020')
    info = mne.create_info(channels, fs, 'eeg', montage=montage)
    if data.ndim == 2:
        fig, ax = plt.subplots(1, data.shape[1])
        #
        for i in range(data.shape[1]):
            mne.viz.plot_topomap(data[:, i],
                                 pos=info, axes=ax[i], show=False)
            ax[i].set(title=f'Spatial Filter {i}')
    else:
        mne.viz.plot_topomap(data, pos=info, show=False)


def plot_spectral_power(filters, samples, fs, kernels):
    '''
    '''
       
    plt.plot()
    for krn in kernels:
        k = filters[:, :, :, :, krn].squeeze()
        p = np.abs(np.fft.fft(k))
        f = np.linspace(0, fs/2, len(p))
        plt.plot(f, p)

    plt.title('Spectral power of kernels')
    plt.xlabel(' Frequency (Hz) ')
    plt.legend(kernels)
    plt.xlim((0, 60))
    plt.show()
