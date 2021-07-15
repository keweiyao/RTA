#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import jv

from scipy.interpolate import interp1d
cs2 = 1./3.
lambda1 = 1#5/7
#E,P,S,T = np.loadtxt("eos.dat").T

#te = interp1d(np.log(E/5.076**3), np.log(T))

def CtauR(tau):
    return (tau/.1)**.3333333333333333

# this is the 2nd-order viscous Bjorken flow
etype='DNMR'
def background(y, tau, L1):
    e, pi = y
    Teff = 1./CtauR(tau)#(e/15)**.25 #np.exp(te(np.log(e)))
    taupi = tauR = 1./Teff 
    pi_eq = tauR*8/(45*tau)*e
    fpi = 1. if etype=='DNMR' else pi/pi_eq
    dedtau = - 4.*e/(3.*tau) + 2*pi/tau
    dpidtau = - (pi-pi_eq)/taupi \
              - pi/tau*(4./3.+2./3.*lambda1*L1*fpi)
    return [dedtau, dpidtau]

def Tabulate(axes, t0,t1,y0,line='-'):
    ax1, ax2 = axes
    ts = np.exp(np.linspace(np.log(t0),np.log(t1),100))



    e1,pi1 = odeint(background, y0, ts, args=(1./3.,) ).T
    e2,pi2 = odeint(background, y0, ts, args=(1,) ).T
    e3,pi3 = odeint(background, y0, ts, args=(3,) ).T

    Teff = (e2/15)**0.25  
    norme1 = e1/e1[-1]*(ts/ts[-1])**(4./3.)
    norme2 = e2/e2[-1]*(ts/ts[-1])**(4./3.)
    norme3 = e3/e3[-1]*(ts/ts[-1])**(4./3.)

    pie1 = .75*pi1/e1
    pie2 = .75*pi2/e2
    pie3 = .75*pi3/e3
 
    ts = ts/CtauR(ts)
    ax1.plot(ts, norme2,'r'+line,alpha=0.6,label=r'$[\pi/(e+P)]_0 = {:1.1f}$'.format(y0[1]/y0[0]/.75))
    #ax1.fill_between(ts,norme1,norme3,color='r',alpha=0.3)

    ax2.plot(ts, pie2,'b'+line,alpha=0.6,label=r'$[\pi/(e+P)]_0 = {:1.1f}$'.format(y0[1]/y0[0]/.75))
    #ax2.fill_between(ts,pie1,pie3,color='b',alpha=0.3)
    
    
if __name__ == "__main__":
    fig, axes = plt.subplots(1,2,figsize=(6.5,3.5), sharex=True)
    e0 = 15
    t0 = 1e-1
    t1 = 1e2
    for pi0, line,file in zip([0,.4, 0.8],['-','-.','--'],
                       ['pi0.dat','pi0d4.dat','pi0d8.dat']):
        file = "RTA_background/"+file
        Tabulate(axes,t0,t1,[e0,pi0*e0*.75],line)
        x,y = np.loadtxt(file).T
        axes[-1].plot(x/CtauR(x),y,line,color='k')
    axes[0].set_ylim(0,1.2)
    #axes[1].set_ylim(1e-2,1e1)
    #axes[1].semilogy()
    axes[0].semilogx()
    axes[0].set_xlim(1e-1,1e1)
    for ax in axes:
        ax.legend(framealpha=0.,loc='best')
        ax.set_xlabel(r"$\tau/\tau_R(\tau)$",fontsize=15)
    axes[0].set_ylabel(r"$e\tau^{4/3} / e_f \tau_f^{4/3}$",fontsize=15)
    axes[1].set_ylabel(r"$\pi/(e+P)$",fontsize=15)
    plt.tight_layout(True)
    plt.suptitle(r"DNMR, $T_{\rm eff} = (T/15)^{1/4}, \tau_\pi=\tau_R=1/T_{\rm eff}, T_{\rm eff,0}= 1$ GeV")
    plt.subplots_adjust(top=.9)
    plt.savefig("DNMR.png", dpi=300)
    plt.show()

