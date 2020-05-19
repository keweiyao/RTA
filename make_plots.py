#!/usr/bin/env python3 
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

fontsmall, fontnormal, fontlarge = 5, 6, 7
offblack = '#262626'
aspect = 1/1.618
resolution = 72.27
textwidth = 320/resolution
textheight = 200/resolution
fullwidth = 300/resolution
fullheight = 200/resolution

plt.rcdefaults()
plt.rcParams.update({
    #'font.family': 'DejaVu Sans',
    #'font.sans-serif': ['Lato'],
    #'mathtext.fontset': 'custom',
    #'mathtext.default': 'it',
    #'mathtext.rm': 'sans',
    #'mathtext.it': 'sans:italic:medium',
    #'mathtext.cal': 'sans',
    'font.size': fontnormal,
    'legend.fontsize': fontsmall,
    'axes.labelsize': fontnormal,
    'axes.titlesize': fontnormal,
    'xtick.labelsize': fontsmall,
    'ytick.labelsize': fontsmall,
    'lines.linewidth': .5,
    'lines.markersize': 3,
    'lines.markeredgewidth': 0,
    'patch.linewidth': .5,
    'axes.linewidth': .4,
    'xtick.major.width': .4,
    'ytick.major.width': .4,
    'xtick.minor.width': .4,
    'ytick.minor.width': .4,
    'xtick.major.size': 1.2,
    'ytick.major.size': 1.2,
    'xtick.minor.size': .8,
    'ytick.minor.size': .8,
    'xtick.major.pad': 1.5,
    'ytick.major.pad': 1.5,
    'axes.formatter.limits': (-5, 5),
    'axes.spines.top': True,
    'axes.spines.right': True,
    'ytick.right': True,
    'axes.labelpad': 3,
    'text.color': offblack,
    'axes.edgecolor': offblack,
    'axes.labelcolor': offblack,
    'xtick.color': offblack,
    'ytick.color': offblack,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'legend.frameon': False,
    'image.cmap': 'Blues',
    'image.interpolation': 'none',
    'pdf.fonttype': 42
})
cm1, cm2 = plt.cm.Blues(.8), plt.cm.Reds(.8)
cb,co,cg,cr = plt.cm.Blues(.6), \
    plt.cm.Oranges(.6), plt.cm.Greens(.6), plt.cm.Reds(.6)
offblack = '#262626'
gray = '0.8'


plotdir = Path('plots')
plotdir.mkdir(exist_ok=True)

plot_functions = {}


def plot(f):
    """
    Plot function decorator.  Calls the function, does several generic tasks,
    and saves the figure as the function name.

    """
    def wrapper(*args, **kwargs):
        logging.info('generating plot: %s', f.__name__)
        f(*args, **kwargs)

        fig = plt.gcf()

        #if not fig.get_tight_layout():
        #    set_tight(fig)

        plotfile = plotdir / '{}.png'.format(f.__name__)
        fig.savefig(str(plotfile), dpi=300)
        logging.info('wrote %s', plotfile)
        plt.close(fig)

    plot_functions[f.__name__] = wrapper

    return wrapper


def set_tight(fig=None, **kwargs):
    """
    Set tight_layout with a better default pad.

    """
    if fig is None:
        fig = plt.gcf()

    kwargs.setdefault('pad', .1)
    fig.set_tight_layout(kwargs)


def auto_ticks(ax, axis='both', minor=False, **kwargs):
    """
    Convenient interface to matplotlib.ticker locators.

    """
    axis_list = []

    if axis in {'x', 'both'}:
        axis_list.append(ax.xaxis)
    if axis in {'y', 'both'}:
        axis_list.append(ax.yaxis)

    for axis in axis_list:
        axis.get_major_locator().set_params(**kwargs)
        if minor:
            axis.set_minor_locator(ticker.AutoMinorLocator(minor))


def Rlm(l0,m0):
    def get_data(l0, m0):
        sre = np.loadtxt("Data/finitek/{:d}{:d}_Re.dat".format(l0,m0)).T
        sim = np.loadtxt("Data/finitek/{:d}{:d}_Im.dat".format(l0,m0)).T
        x = sre[0]
        y = sre[1:]+1j*sim[1:]
        Alm = {}
        Lmax = int(np.sqrt(len(y))-1)
        for l in np.arange(0, Lmax+1):
            Alm[l] = {}
            for m in np.arange(-l, l+1):
                Alm[l][m] = y[(l**2+(m+l)).astype(int)]
        return x, Lmax, Alm
    x, Lmax, Alm = get_data(l0, m0)

    fig, axes = plt.subplots(ncols=Lmax+1, nrows=2, figsize=(textwidth, textwidth*.45), sharex=True, sharey=True)
    for l, axc in zip(np.arange(0, Lmax+1), axes.T):
        ax1, ax2 = axc
        Nre = Nim = 0
        for m in np.arange(-l, l+1):
            color = plt.cm.winter((m+l+0.5)/(2*l+1))
            y = Alm[l][m]
            if (np.abs(y.real)).max()>1e-5:
                Nre += 1
                ax1.plot(x, y.real, '-', color=color, label=r"$m={:d}$".format(m), alpha=1)
            if (np.abs(y.imag)).max()>1e-5:
                Nim += 1
                ax2.plot(x, y.imag, '-', color=color, label=r"$m={:d}$".format(m), alpha=1)
        if l==0:
            ax1.set_ylabel(r"$\mathfrak{Re}A_{lm}$")
            ax2.set_ylabel(r"$\mathfrak{Im}A_{lm}$")
        if Nre>0:
            ax1.legend()
        if Nim>0:
            ax2.legend()
        ax1.set_title(r"$l={:d}$".format(l))
        ax2.set_xticks([-1,0,1])
        ax2.set_xticklabels(["$0.1$","$1$","$10$"])
        ax2.set_xlabel(r"$\tau/\tau_R$")
        ax2.set_xlim(-1.1, 1.3)
        ax2.set_ylim(-1,1)
    plt.subplots_adjust(wspace=.05, hspace=.05, top=.83, left=.08, right=.99, bottom=.15)
    plt.suptitle(r"$l_0={:d}, m_0={:d}$".format(l0,m0)+r", $(\eta/s)_{\mathrm{eff}=0.2}, \kappa=1, k=1$ GeV")

@plot
def R00_lm():
    Rlm(0,0)
@plot
def R10_lm():
    Rlm(1,0)
@plot
def R11_lm():
    Rlm(1,1)
@plot
def R1m1_lm():
    Rlm(1,-1)
@plot
def R22_lm():
    Rlm(2,2)
@plot
def R21_lm():
    Rlm(2,1)
@plot
def R20_lm():
    Rlm(2,0)
@plot
def R2m1_lm():
    Rlm(2,-1)
@plot
def R2m2_lm():
    Rlm(2,-2)
    
if __name__ == '__main__':
    import argparse

    choices = list(plot_functions)

    def arg_to_plot(arg):
        arg = Path(arg).stem
        if arg not in choices:
            raise argparse.ArgumentTypeError(arg)
        return arg

    parser = argparse.ArgumentParser(description='generate plots')
    parser.add_argument(
        'plots', nargs='*', type=arg_to_plot, metavar='PLOT',
        help='{} (default: all)'.format(', '.join(choices).join('{}'))
    )
    args = parser.parse_args()

    if args.plots:
        for p in args.plots:
            plot_functions[p]()
    else:
        for f in plot_functions.values():
            f()
