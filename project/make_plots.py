import matplotlib.pyplot as plt
import numpy as np
from my_mcmc import gaussian_2d_ind

def set_plot_params(nrows=1, ncols=1, figsize=None):
    """
    Functions to create nicer looking plots
    """

    plt.rc('font',**{'family':'STIXGeneral'})
    plt.rc('text', usetex=False)

    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)

    plt.rc('lines', linewidth=2)

    if figsize is None:
        fig, ax = plt.subplots(nrows,ncols, figsize=(ncols*5 + (ncols)*3,nrows*5+(nrows-1)*3))
    else:
         fig, ax = plt.subplots(nrows,ncols, figsize=figsize)

    if type(ax)==type(np.zeros(1)):
        for a in ax.ravel():
            set_ticks(a)
    else:
        set_ticks(ax)

    return fig, ax

def set_ticks(ax):
    ax.tick_params('both', which='minor', length=4, direction='in', bottom=True, top=True, left=True, right=True)
    ax.tick_params('both', which='major', length=8, direction='in', bottom=True, top=True, left=True, right=True)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    return


def param_histograms(xchain, x0, nparams=5, labels=['$\mu_{x}$','$\mu_{y}$','$\sigma^{2}_{x}$','$\sigma^{2}_{y}$', 'A'], figsize=(10,15)):
    fig, ax = set_plot_params(nparams,figsize=figsize)
    if len(labels)!= nparams:
        return "List of labels length must equal number of parameters"
    for i in range(nparams):
        ax[i].hist(xchain[:,i],bins=50, label=labels[i], color='lavender')
        ax[i].legend()
        ax[i].axvline(np.average(xchain, axis=0)[i], linestyle='--', color='gray')
        ax[i].axvline(x0[i], color='black')
    return


def time_series(xchain, labels = ['$\mu_{x}$','$\mu_{y}$','$\sigma^{2}_{x}$','$\sigma^{2}_{y}$', 'A'], figsize=(10,12)):
    fig, ax = set_plot_params(5,1, figsize=figsize)
    for i in range(5):
        ax[i].plot(range(len(xchain[:,i])), xchain[:,i], color='black', label=labels[i])
        ax[i].legend(loc='upper right')
    return

def autocorr(xchain, lag, labels=['$\mu_{x}$','$\mu_{y}$','$\sigma^{2}_{x}$','$\sigma^{2}_{y}$', 'A']):
    fig, ax = set_plot_params(5, figsize=(10,10))
    for i in range(5):
        ax[i].scatter(xchain[lag:,i], xchain[:-lag,i], label=labels[i], color='black')
    return
        

def compare_model(img, xchain, t_vals, model_input):
    plt.subplot(121)
    plt.imshow(img, origin='lower', cmap='gray', vmax=img.max())
    plt.title('Data')
    plt.subplot(122)
    plt.title('Model')
    model = model_input(t_vals, x=np.mean(xchain, axis=0))
    plt.imshow(model, origin='lower', cmap='gray', vmax=img.max())
    plt.show()
    return 