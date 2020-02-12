import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


def plot_temporal_filters(filters, fs):
    '''
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
                col.plot(time, filters[:,:,:,:,k].squeeze())
                col.set_title(f'Temporal Filter {k+1}')
                k+=1

    fig.tight_layout()
    plt.xlabel(' Time (ms) ')
    plt.show()


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def plot_topomaps(data=None, channels=None):
    '''
    '''
    fpath = '/content/drive/My Drive/data/ssvep_benchmark_dataset/64-channels.txt'
    f = open(fpath, 'r')
    locs = f.read()
    a = locs.split('\n')
    pol = [np.array(i.split('\t')[1:3]).astype(float) for i in a if i]
    pol = np.array(pol)
    chs = [i.split('\t')[-1].strip() for i in a if i]
    pol[:,0] = pol[:,0] * (np.pi/180) 
    x,y = pol2cart(pol[:,1], pol[:,0])  
    #
    interp_detail = 100
    interpx = np.linspace(x.min()-.2, x.max()+.25, interp_detail)
    interpy = np.linspace(y.min(), y.max(), interp_detail)

    gridx, gridy = np.meshgrid(interpx, interpy)

    dat = np.random.uniform(-2,2,64)
    triang = tri.Triangulation(y, x)
    interpolator = tri.LinearTriInterpolator(triang, dat)
    zi = interpolator(gridx, gridy)

    an = np.linspace(0, 2 * np.pi, 100)
    plt.plot(.8*np.cos(an), .8*np.sin(an),color='black')
    plt.scatter(y,x,marker='o',c='black')
    plt.contourf(interpy, interpx, zi, cmap="RdBu_r", alpha=.5)
    #plt.xlim([-.6, .6])
    #plt.ylim([-.6,.6])
    plt.title('Electrode locations')
    plt.colorbar()

    plt.show() 