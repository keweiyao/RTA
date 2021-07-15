#!/usr/bin/env python3 
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import interpn, interp1d
from scipy.special import jv

fontsmall, fontnormal, fontlarge = 6, 6.5, 7
offblack = '#262626'
aspect = 1/1.618
resolution = 72.27
textwidth = 350/resolution
textheight = 300/resolution
fullwidth = 300/resolution
fullheight = 200/resolution

plt.rcdefaults()
plt.rcParams.update({
    'font.size': fontnormal,
    'legend.fontsize': fontsmall,
    'axes.labelsize': fontnormal,
    'axes.titlesize': fontnormal,
    'xtick.labelsize': fontsmall,
    'ytick.labelsize': fontsmall,
    'lines.linewidth': .7,
    'lines.markersize': 3,
    'lines.markeredgewidth': 0,
    'patch.linewidth': 1,
    'axes.linewidth': .6,
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
cdb = plt.cm.Blues(.8),
offblack = '#262626'
gray = '0.8'


plotdir = Path('Newplots')
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

        if not fig.get_tight_layout():
            set_tight(fig)

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
import h5py
@plot
def G_vs_k():
    
    fig, axes = plt.subplots(3,5,figsize=(textwidth, .6*textwidth), sharex=True, sharey=True)
    f1 = h5py.File("OfficialData/Kinetic_Response_vs_k.h5",'r') 
    f3 = h5py.File("OfficialData/IS_Response_vs_k.h5",'r') 
    f2 = h5py.File("OfficialData/NS_Response_vs_k.h5",'r') 
    f4 = h5py.File("OfficialData/MISstar_Response_vs_k.h5",'r') 
    for j, (t0,t1) in enumerate([(1,2),(1,4),(1,8)]):
        gname = "t{}-{}".format(t0,t1)
        for i, ((l0,m0),(l1,m1),name,title,P) in enumerate(zip(
           [(0,0), (0,0), (1,1), (1,1), (1,-1)],
           [(0,0), (1,1), (0,0), (1,1), (1,-1)],
           ['e2e', 'e2p', 'p2e', 'p2p', 's2s'],
           [r'$G_s^s$', r'$G_s^v$', r'$G_v^s$', r'$G_v^v$', r'$G^\delta$'],
           [4./3., 4./3., 1, 1., 1.]
          )):
            scale = (t1/t0)**P

            ax = axes[j,i]
            ax.set_xlim(0,10)
            ax.set_ylim(-1.5, 1.8)

            x1 = f1[gname+'/'+name+'/k'][()]
            y1 = f1[gname+'/'+name+'/Gk'][()].imag if name in ['e2p','p2e'] else f1[gname+'/'+name+'/Gk'][()].real
            ax.plot(x1, y1*scale, 'k.', label='Kinetic' if j==2 and i==3 else '')

            x2 = f2[gname+'/'+name+'/k'][()]
            y2 = f2[gname+'/'+name+'/Gk'][()].imag if name in ['e2p','p2e'] else f2[gname+'/'+name+'/Gk'][()].real
            ax.plot(x2, y2*scale, 'r--', label=r'$1^{\rm st}$' if j==2 and i==3 else '')

            x3 = f3[gname+'/'+name+'/k'][()]
            y3 = f3[gname+'/'+name+'/Gk'][()].imag if name in ['e2p','p2e'] else f3[gname+'/'+name+'/Gk'][()].real
            ax.plot(x3, y3*scale, 'b-.', label=r'$2^{\rm nd}$' if j==2 and i==4 else '')

            x4 = f4['d0.13pi1/'+gname+'/'+name+'/k'][()]
            y4 = f4['d0.13pi1/'+gname+'/'+name+'/Gk'][()].imag if name in ['e2p','p2e'] else f4['d0.13pi1/'+gname+'/'+name+'/Gk'][()].real
            ax.plot(x4, y4*scale, 'g-', label=r'MIS${}^*$' if j==2 and i==4 else '')

            if ax.is_last_row():
                ax.set_xlabel(r"$k\tau_R$")
            if ax.is_first_row():
                ax.set_title(title)
            if ax.is_first_col():
                ax.set_ylabel(r"$G(k) (\tau/\tau_0)^P$")
            ax.annotate(r"$\tau_0={}, \tau={}$".format(t0,t1), xy=(.1,.85), xycoords="axes fraction")
        axes[-1,-2].legend(loc='lower right')
        axes[-1,-1].legend(loc='lower right')
            
from scipy.special import jv
def k2r(k, Gk, channel, dtau):
    r = np.linspace(0,2*dtau,100)
    kr = np.outer(r, k)
    dk = k[1]-k[0]
    if channel == 'e2e':
        Gr = np.sum(jv(0, kr)*k*Gk*np.exp(-k**2/200), axis=1)*dk/2/np.pi
    if channel == 'e2p':
        Gr = np.sum(jv(1, kr)*k*Gk*np.exp(-k**2/200), axis=1)*dk/2/np.pi
    if channel == 'p2e':
        Gr = np.sum(jv(1, kr)*k*Gk*np.exp(-k**2/200), axis=1)*dk/2/np.pi
    if channel == '+':
        Gr = np.sum(jv(0, kr)*k*Gk*np.exp(-k**2/200), axis=1)*dk/2/np.pi
    if channel == '-':
        Gr = - np.sum(jv(2, kr)*k*Gk*np.exp(-k**2/200), axis=1)*dk/2/np.pi
    return r, Gr

@plot
def G_vs_r():
    fig, axes = plt.subplots(3,5,figsize=(textwidth, .6*textwidth), sharex=True, sharey=True)
    f1 = h5py.File("OfficialData/Kinetic_Response_vs_k.h5",'r')
    f2 = h5py.File("OfficialData/NS_Response_vs_k.h5",'r')  
    f3 = h5py.File("OfficialData/IS_Response_vs_k.h5",'r') 

    f4 = h5py.File("OfficialData/MISstar_Response_vs_k.h5",'r') 
    for j, (t0,t1) in enumerate([(1,2),(1,4),(1,8)]):
        gname0 = "t{}-{}".format(t0,t1)
        for i, ((l0,m0),(l1,m1),name,title) in enumerate(zip(
           [(0,0), (0,0), (1,1), (1,1), (1,-1)],
           [(0,0), (1,1), (0,0), (1,1), (1,-1)],
           ['e2e', 'e2p', 'p2e', 'p2p', 's2s'],
           [r'$G_s^s$', r'$G_v^s$', r'$G_s^v$', r'$G_v^v$', r'$G^\delta$']
          )):
            dt = t1-t0
            ax = axes[j,i]
            ax.set_xlim(0,1.8)
            ax.set_ylim(-3,3)

            for f, fmt, (j0,i0),label,gname in zip([f1,f4,f4,f4], ['k.','r--','b-.','g-'], 
                                       [(0,3),(0,3),(0,4),(0,4)], 
                                       ['Kinetic', r'$1^{\rm st}$',r'$2^{\rm nd}$', r"MIS${}^*$"],
                          [gname0, 'd0.0pi1/'+gname0, 'd1.0pi1/'+gname0, 'd0.13pi1/'+gname0]):
                #if fmt=='b-.':
                #    if  j!=0 or i!=0:
                #        continue
                x1 = f[gname+'/'+name+'/k'][()]
                if name in ['e2p','p2e']:
                    y1 = (-1j*f[gname+'/'+name+'/Gk'][()]).real  
                    r, Gr = k2r(x1, y1, name, dt)
                if name == 'e2e':
                    y1 = f[gname+'/'+name+'/Gk'][()].real
                    r, Gr = k2r(x1, y1, name, dt)
                if name in ['p2p', 's2s']:
                    yp = f[gname+'/p2p/Gk'][()].real + f[gname+'/s2s/Gk'][()].real
                    ym = f[gname+'/p2p/Gk'][()].real - f[gname+'/s2s/Gk'][()].real
                    r, Gp = k2r(x1, yp, '+', dt)
                    r, Gm = k2r(x1, ym, '-', dt)
                    if name=='p2p':
                        Gr = (Gp+Gm)/2.
                    else:
                        Gr = (Gp-Gm)/2.
                ax.plot(r/dt, r*(t1/t0)**2*Gr, fmt, label=label if j==j0 and i==i0 else '')

                ax.set_xticks([0,.5,1.,1.5])

                ax.legend(loc='lower right')

            if ax.is_last_row():
                ax.set_xlabel(r"$r/\Delta\tau$")
            if ax.is_first_row():
                ax.set_title(title)
            if ax.is_first_col():
                ax.set_ylabel(r"$r G(r) (\tau/\tau_0)^2$")
            ax.annotate(r"$\tau_0={}, \tau={}$".format(t0,t1), xy=(.1,.85), xycoords="axes fraction")

    set_tight(fig,rect=[.03,.03,.99,.99])            


def jetk2r(ts, ks, Gee, Gpe, Gep, Gpp, Gtt):
    dk = ks[1]-ks[0]
    def C(t):
        T = 1/t**(1./3.)
        return T**2
    smear = np.exp(-ks**2/2/4**2)
    Nr = 75
    Nphi = 150
    print(1.5*(ts[-1]-ts[0]))
    rs = np.linspace(1e-9, 9, Nr)
    phis = np.linspace(0, 2*np.pi, Nphi)
    v = 1.0
    e = np.zeros([Nr, Nphi])
    px = np.zeros([Nr, Nphi])
    py = np.zeros([Nr, Nphi])
    dt = ts[1:]-ts[:-1]
    for it, (t, dt) in enumerate(zip(ts[1:], dt)):
        print(it, t)
        for i, r in enumerate(rs):
            for j, phi in enumerate(phis):
                Ry = r*np.sin(phi)
                Rx = r*np.cos(phi)-v*(t-ts[0])
                R = np.sqrt( Rx**2 + Ry**2 )
                kR = ks*R+1e-9
                jv0 = jv(0, kR)
                jv1 = jv(1, kR)
                jv2 = jv(2, kR)
                phiR = np.arctan2(Ry, Rx)
                e[i,j] += dt * C(t) * (ks*smear * (
                         jv0 * Gee[it].real
                       - jv1 * np.cos(phiR) * Gpe[it].imag
                       )).sum()
                px[i,j] += dt * C(t) * (ks*smear * (
                         jv0 * ( Gpp[it].real * np.cos(phiR)**2
                                     + Gtt[it].real * np.sin(phiR)**2 )
                       - jv1 * np.cos(phiR) * Gep[it].imag
                      - jv1/kR * np.cos(2*phiR) * (Gpp[it] - Gtt[it])
                       )).sum()
                py[i,j] += dt * C(t) * (ks*smear * (
                    jv1 * Gep[it].imag * np.sin(phiR)
                 - (jv1/kR - jv0/2) * np.sin(2*phiR) 
                              * (Gpp[it] - Gtt[it])
                 )).sum()
    e *= dk/(2*np.pi)
    px *= dk/(2*np.pi)
    py *= dk/(2*np.pi)
    return rs, phis, rs*e.T, rs*px.T, rs*py.T

def jet(t0, t1):
    fig, axes = plt.subplots(1,3,subplot_kw={'projection': 'polar'},figsize=(.8*textwidth, .37*textwidth), sharex=True, sharey=True )
    name = 'd0d0pi1/t{}-{}/'.format(t0,t1)
    f1 = h5py.File("OfficialData/MISstar_Response_vs_t_k.h5",'r')
    Gee = f1[name+'/e2e/Gk'][()]
    Gep = f1[name+'/e2p/Gk'][()]
    Gpe = f1[name+'/p2e/Gk'][()]
    Gpp = f1[name+'/p2p/Gk'][()]
    Gtt = f1[name+'/s2s/Gk'][()]
    t = f1[name+'/e2e/t'][()]
    t = np.linspace(t0,t1,21)
    k = f1[name+'/e2e/k'][()]

    print(t.shape, Gtt.shape, Gpp.shape,  Gpe.shape, Gep.shape,  Gee.shape)
    r, theta, e, px, py = jetk2r(t, k, Gee, Gpe, Gep, Gpp, Gtt)
    r, th = np.meshgrid(r, theta)

    for ax, it, name in zip(axes, [e, px, py], 
                           [r"$\delta e$", r"$\delta p_x$", r"$\delta p_y$"]):
        ax.pcolormesh(th, r, it)
        ax.set_title(name)
        ax.set_xticklabels([])
        ax.set_yticks([0,4,8])
  
        ax.set_ylim(0,9)
    #set_tight(fig, rect=[.005, .995, .01, .98])


@plot
def jet_1_2():
    jet(1,2)

@plot
def jet_1_4():
    jet(1,4)

@plot
def jet_1_8():
    jet(1,8)


def jet3Dk2r(ts, ks, kappas, Gee, Gpe, Gep, Gpp, Gtt, Gel, Gpl):
    dk = ks[1]-ks[0]
    dkappa = kappas[1]-kappas[0]
    def C(t):
        T = 1/t**(1./3.)
        return T**2

    # perform the eta integration first
    eta_smear = np.exp(-.5*kappas**2/4**2)

    smear = np.exp(-.5*ks**2/4**2)
    Nr = 40
    Nphi = 60
    Neta = 41
    rs = np.linspace(1e-9, 6, Nr)
    phis = np.linspace(0, 2*np.pi, Nphi)
    etas = np.linspace(-4,4,Neta)
    v = 1.0
    e = np.zeros([Nr, Nphi, Neta])
    px = np.zeros([Nr, Nphi, Neta])
    py = np.zeros([Nr, Nphi, Neta])
    peta = np.zeros([Nr, Nphi, Neta])
    dt = ts[1:]-ts[:-1]
    for it, (t, dt) in enumerate(zip(ts[1:], dt)):
        print(it, t)
        for i, r in enumerate(rs):
            for j, phi in enumerate(phis):
                Ry = r*np.sin(phi)
                Rx = r*np.cos(phi)-v*(t-ts[0])
                R = np.sqrt( Rx**2 + Ry**2 )
                kR = ks*R+1e-9
                jv0 = jv(0, kR)
                jv1 = jv(1, kR)
                jv2 = jv(2, kR)
                phiR = np.arctan2(Ry, Rx)
                for l, eta  in enumerate(etas):
                    e[i,j,l]  += dt * C(t) * (np.sum(ks*smear * (jv0 * Gee[it].real.T- jv1 * np.cos(phiR) * Gpe[it].imag.T), axis=1) 
                                               * eta_smear * 2*np.cos(eta*kappas) ).sum()
                    px[i,j,l] += (np.sum(dt * C(t) * (ks*smear * (
                         jv0 * ( Gpp[it].real.T * np.cos(phiR)**2
                               + Gtt[it].real.T * np.sin(phiR)**2 )
                       - jv1 * np.cos(phiR) * Gep[it].imag.T
                       - jv1/kR * np.cos(2*phiR) * (Gpp[it].real.T - Gtt[it].real.T)
                        )), axis=1) * eta_smear * 2*np.cos(eta*kappas) ).sum()
                    py[i,j,l] += (np.sum(dt * C(t) * (ks*smear * (
                        jv1 * Gep[it].imag.T * np.sin(phiR)
                     - (jv1/kR - jv0/2) * np.sin(2*phiR) 
                              * (Gpp[it].real.T - Gtt[it].real.T)
                        )), axis=1) * eta_smear * 2*np.cos(eta*kappas) ).sum()
                    peta[i,j,l]  += (np.sum(dt * C(t) * (ks*smear * (
                         jv0 * Gel[it].T
                       + 1j*jv1 * np.cos(phiR) * Gpl[it].T
                        )), axis=1).real * eta_smear * 2*np.sin(eta*kappas) ).sum()
    
    e  *= dk/(2*np.pi) * dkappa/(2*np.pi)
    px *= dk/(2*np.pi) * dkappa/(2*np.pi)
    py *= dk/(2*np.pi) * dkappa/(2*np.pi)
    peta *= dk/(2*np.pi) * dkappa/(2*np.pi)

    for ir, rv in enumerate(rs):
        e[ir] *= rv
        px[ir] *= rv
        py[ir] *= rv
        peta[ir] *= rv

    return rs, phis, etas, e, px, py, peta

def jet3D(t0, t1):
    fig, axes = plt.subplots(1,2,figsize=(.6*textwidth, .3*textwidth))
    name = 't{}-{}/'.format(t0,t1)
    f1 = h5py.File("OfficialData/Kinetic_3D_vs_t_k.h5",'r')
    Gee = f1[name+'/e2e/Gk'][()]
    Gep = f1[name+'/e2p/Gk'][()]
    Gpe = f1[name+'/p2e/Gk'][()]
    Gpp = f1[name+'/p2p/Gk'][()]
    Gtt = f1[name+'/s2s/Gk'][()]
    Gel = f1[name+'/e2l/Gk'][()]
    Gpl = f1[name+'/p2l/Gk'][()]
    t = f1[name+'/e2e/t'][()]
    k = f1[name+'/e2e/k'][()]
    kappa = f1[name+'/e2e/kappa'][()]

    r, theta, etas, e, px, py, peta = jet3Dk2r(t, k, kappa, Gee, Gpe, Gep, Gpp, Gtt, Gel, Gpl)
    dr = r[1]-r[0]
    dtheta = theta[1]-theta[0]
    for ax, it in zip(axes, [e,px]):
        ax.plot(etas, np.sum(it[:,:,:], axis=(0,1))*dr*dtheta,'r-')
    
@plot
def jet3D_1_4():
    jet3D(1,4)


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
