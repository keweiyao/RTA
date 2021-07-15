#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import jv

from scipy.interpolate import interp1d
cs2 = 1./3.
lambda1 = 1
tauR0 = 1
cs2 = 1./3.
one_and_cs2 = 1. + cs2
f = np.load("Kinetic-Background.npy")
ibg = 1


t = f['t']
e = f['e'][ibg]
pi = f['pi'][ibg]
BG_e = interp1d(np.log(t), np.log(e), fill_value='extrapolate')
BG_piovere = interp1d(np.log(t), pi/e, fill_value='extrapolate')

tm = (t[1:] + t[:-1])/2.
dt = (t[1:] - t[:-1])
de = (e[1:] - e[:-1])
em = (t[1:] + t[:-1])/2.
dpi = (pi[1:] - pi[:-1])
BG_dlne_dlnt = interp1d(np.log(tm), de/dt*tm/em, fill_value='extrapolate')
BG_inve_dpi_dlnt = interp1d(np.log(tm), dpi/dt*tm/em, fill_value='extrapolate')

def GetTauR(tau0, tau):
    return tauR0

def compute_pi(tau, y, tau0, kx, ky, kappa, l1, g, piscale):
    tauR = GetTauR(tau0, tau)
    gammas = tauR/5.
    cg = 1.-g
    taupi = tauR*piscale
    e, gx, gy, geta, L200, L210, L211, L220, L221 = y
    E = np.exp(BG_e(np.log(tau)))
    Pi = BG_piovere(np.log(tau)) * E
    P = E/3.
    H = E + P

    DSigma_200 = 1j*(kappa*geta/tau - .5*(kx*gx+ky*gy))
    DSigma_210 = 1j*(kx*geta + kappa/tau*gx)
    DSigma_211 = 1j*(ky*geta + kappa/tau*gy)
    DSigma_220 = 1j*(kx*gx - ky*gy)
    DSigma_221 = 1j*(kx*gy + ky*gx)
    DSigma_000 = 1j*(kx*gx + ky*gy + kappa/tau*geta)
    # NS limit of pi
    NS200 = -2*DSigma_200 * (gammas +    taupi*lambda1*l1*Pi/H)
    NS210 = -2*DSigma_210 * (gammas + .5*taupi*lambda1*l1*Pi/H)
    NS211 = -2*DSigma_211 * (gammas + .5*taupi*lambda1*l1*Pi/H)
    NS220 = -2*DSigma_220 * (gammas -    taupi*lambda1*l1*Pi/H)
    NS221 = -2*DSigma_221 * (gammas -    taupi*lambda1*l1*Pi/H)
    pixyeq = .5*NS221
    pixzeq = NS210
    piyzeq = NS220
    pizzeq = NS200*2./3.
    pixxeq =   0.5*NS220 - NS200/3.
    piyyeq = - 0.5*NS220 - NS200/3.
    #second order like pi and dpi/dtau
    pixy = .5*L221
    pixz = L210
    piyz = L220
    pizz = L200*2./3.
    pixx =   0.5*L220 - L200/3.
    piyy = - 0.5*L220 - L200/3.
    D200 = - (L200-NS200*cg)/taupi - (4./3. + 2*lambda1*l1/3.)/tau*L200 \
           - e/E*2*gammas*H/tau/taupi
    D210 = - (L210-NS210*cg)/taupi - (4./3. +   lambda1*l1/3.)/tau*L210
    D211 = - (L211-NS211*cg)/taupi - (4./3. +   lambda1*l1/3.)/tau*L211
    D220 = - (L220-NS220*cg)/taupi - (4./3. - 2*lambda1*l1/3.)/tau*L220
    D221 = - (L221-NS221*cg)/taupi - (4./3. - 2*lambda1*l1/3.)/tau*L221
    # combine the two
    return g*pixxeq+cg*pixx, g*pixyeq+cg*pixy, g*pixzeq+cg*pixz, \
           g*piyyeq+cg*piyy, g*piyzeq+cg*piyz, g*pizzeq+cg*pizz, \
           D200, D210, D211, D220, D221

def Perturbation_consrvation_equation(tau, y, tau0, kx, ky, kappa, l1, gscale, piscale):
    e, gx, gy, geta, L200, L210, L211, L220, L221 = y
    E = np.exp(BG_e(np.log(tau)))
    Pi = BG_piovere(np.log(tau)) * E
    P = E/3.
    H = (E+P)
    pixx, pixy, pixz, piyy, piyz, pizz,\
    D200, D210, D211, D220, D221 = compute_pi(tau, y, tau0, kx, ky, kappa, l1, gscale, piscale)
    dEdtau, dPidtau = E/tau*BG_dlne_dlnt(np.log(tau)), E/tau*BG_inve_dpi_dlnt(np.log(tau))
    dNdtau = (E*dPidtau - Pi*dEdtau)/H**2
    PG_200 = 1j*(kappa*geta/tau - (kx*gx+ky*gy)/2.)
    De  = - one_and_cs2/tau*e - pizz/tau - 1j*(kx*gx+ky*gy+kappa*geta/tau) - 2*Pi/H*PG_200 
    Dgx = - (1/tau-dNdtau)*gx - cs2*1j*kx*e -1j*(kx*pixx + ky*pixy + kappa/tau*pixz)
    Dgy = - (1/tau-dNdtau)*gy - cs2*1j*ky*e -1j*(kx*pixy + ky*piyy + kappa/tau*piyz)
    Dgeta = - (2./tau+2*dNdtau)*geta - cs2*1j*kappa/tau*e -1j*(kx*pixz + ky*piyz + kappa/tau*pizz)
    Dgx /= (1-Pi/H)
    Dgy /= (1-Pi/H)
    Dgeta /= (1+2*Pi/H)
    return np.array([De, 
                     Dgx, Dgy, Dgeta, 
                     D200, D210, D211, D220, D221])


def MIS_star(Y, tau, tau0, 
             kx, ky, kappa, 
             l1, delta_plus, piscale):
    y = Y[:9] + 1j*Y[9:]
    Dy = Perturbation_consrvation_equation(tau, y, tau0, kx, ky, kappa, l1, delta_plus, piscale)
    return np.concatenate([Dy.real, Dy.imag])

def plot_Gk_tau(t0=1e-1, t1=1e1):
    fig, ax = plt.subplots(1,1,figsize=(5,4), sharex=True)
    e0 = 15

    ts = np.exp(np.linspace(np.log(t0), np.log(t1), 10000))
    
  
    y0 = np.array([1, 
                   0, 0, 0, 
                   0, 0, 0, 0, 0])
    
    for ik, k in enumerate([0, 1, 2, 4]):      
        params = (t0, k, 0, 0, 7/5, 1e-8, 1)
        res = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params).T
        res = res[:9]+1j*res[9:] 
        color = plt.cm.coolwarm(ik/3)
        ax.plot(ts/GetTauR(t0, ts), np.abs(res[0]), color=color, label=r"$k\tau_{{R,0}}={}$".format(k), lw=2)
        
    ax.semilogx()
    ax.semilogy() 
    ax.legend()
    ax.set_xlabel(r"$\tau/\tau_R(\tau)$",fontsize=15)
    ax.set_ylabel(r"$|G_{s}^{s}(\tau', \tau)|$",fontsize=15)
    plt.tight_layout(True)
    plt.subplots_adjust(top=.9)
    plt.savefig("Gk_ee_tau.png", dpi=300)
    plt.show()


def plot_Gk_k(t0=.1, t1=5):
    fig, ax = plt.subplots(1,1,figsize=(5,4), sharex=True)
    e0 = 15

    ts = np.exp(np.linspace(np.log(t0), np.log(t1), 100))
    
  
    y0 = np.array([1, 
                   0, 0, 0, 
                   0, 0, 0, 0, 0])
    ks = np.linspace(0,10,100)

    x, y = np.loadtxt("KineticResponse/Gs_k_tau1_5.dat".format(t0,t1)).T
    ax.plot(x,y*(t1/t0)*1.3333,'k-', label='Kinetic')

    

    res = []
    for k in ks:
        params = (t0, k, 0, 0, 1, 1-1e-9, 1)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params)[-1]
        d = d[:9]+1j*d[9:] 
        res.append(d[0])
    res = np.array(res)

    ax.plot(ks, res*(t1/t0)**1.3333, 'k--', label='NS')
  
    res = []
    for k in ks:
        params = (t0, k, 0, 0, 1, 1e-9, 1)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params)[-1]
        d = d[:9]+1j*d[9:] 
        res.append(d[0])
    res = np.array(res)

    ax.plot(ks, res*(t1/t0)**1.3333, color=plt.cm.Reds(.5), lw=2, 
            label=r'IS, $\lambda_1=5/7, \delta=0,\tau_\pi=\tau_R$')


    res = []
    for k in ks:
        params = (t0, k, 0, 0, 1, 0.12, 1)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params)[-1]
        d = d[:9]+1j*d[9:] 
        res.append(d[0])
    res = np.array(res)

    ax.plot(ks, res*(t1/t0)**1.3333, color=plt.cm.Greens(.5), lw=2, 
            label=r'IS, $\lambda_1=5/7, \delta=0.12, \tau_\pi=\tau_R$')

    res = []
    for k in ks:
        params = (t0, k, 0, 0, 1, 0.12, .5)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params)[-1]
        d = d[:9]+1j*d[9:] 
        res.append(d[0])
    res = np.array(res)

    ax.plot(ks, res*(t1/t0)**1.3333, color=plt.cm.Blues(.5), lw=2, 
            label=r'IS, $\lambda_1=5/7, \delta=0.12, \tau_\pi=\tau_R/2$')
 
    ax.legend()
    ax.plot([0,10],[0,0],'k-')
    ax.set_xlabel(r"$k_s$",fontsize=15)
    ax.set_ylabel(r"$G_s(\tau', \tau; k)$",fontsize=15)
    plt.tight_layout(True)
    plt.subplots_adjust(top=.9)
    plt.savefig("Gss_k.png", dpi=300)
    plt.show()

def plot_background_solution():
    plt.figure(figsize=(7,3.5))
    for ibg, color, xi, l in zip([0,1,2], 'rgb', [1./3., 1, 3.], ['1/3','1','3']):
        t = f['t']
        e = f['e'][ibg]
        pi = -f['pi'][ibg]
        BG_e = interp1d(np.log(t), np.log(e), fill_value='extrapolate')
        BG_piovere = interp1d(np.log(t), pi/e, fill_value='extrapolate')

        tau = t
        plt.subplot(1,2,1)
        plt.plot(tau, (tau/tau[0])**(4./3.)*np.exp(BG_e(np.log(tau)))/15, label=r"$\xi = {}$".format(l))
        plt.subplot(1,2,2)
        plt.plot(tau, (e/3+2*pi)/(e/3-pi), label=r"$\xi = {}$".format(l))
    plt.subplot(1,2,1)
    plt.semilogx()
    plt.legend()
    plt.ylim(0,3)
    plt.xlabel(r"$\tau/\tau_R$", fontsize=15)
    plt.ylabel(r"$(\tau/\tau_0)^{4/3}e/e_0$", fontsize=15)

    plt.subplot(1,2,2)
    plt.semilogx()
    plt.legend()
    plt.xlabel(r"$\tau/\tau_R$", fontsize=15)
    plt.ylabel(r"$P_L/P_T$", fontsize=15)
    plt.ylim(0,2.5)
    plt.tight_layout(True)
    plt.savefig("Background.png", dpi=300)

def Response_vs_k(tau0, tau1, Ls, Ms, Lr, Mr):
    ids = {(0,0):0,(1,1):1,(1,-1):2,(1,0):3}[(Ls, Ms)]
    idr = {(0,0):0,(1,1):1,(1,-1):2,(1,0):3}[(Lr, Mr)]
    ks = np.linspace(0,30,300)
    G = np.zeros_like(ks, dtype=np.complex)
    for ik, k in enumerate(ks):
        if ik % 10==0:
            print(tau0, tau1, name, ik)
        y0 = np.zeros(9, dtype=np.complex)
        y0[ids] = 1
        params = (tau0, k, 0, 0, 1, 1e-16, 1)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), [tau0,tau1], args=params)[-1]
        d = d[:9]+1j*d[9:] 
        G[ik] = d[idr]
    return ks, G

def Response_vs_t(tau0, tau1, Ls, Ms, Lr, Mr):
    ids = {(0,0):0,(1,1):1,(1,-1):2,(1,0):3}[(Ls, Ms)]
    idr = {(0,0):0,(1,1):1,(1,-1):2,(1,0):3}[(Lr, Mr)]
    ks = [0, 0.5, 1.0, 1.5, 2.0]
    ts = np.linspace(tau0, tau1, 101)
    G = np.zeros([ks.shape, ts.shape], dtype=np.complex)
    for ik, k in enumerate(ks):
        print(tau0, tau1, name, ik)
        y0 = np.zeros(9, dtype=np.complex)
        y0[ids] = 1
        params = (tau0, k, 0, 0, 1, 0.13, 1)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params)
        d = d[:9]+1j*d[9:] 
        G[ik] = d[idr]
    return ks, ts, G


import sys
import h5py
if sys.argv[1]=='1':
    with h5py.File("OfficialData/MISstar_Response_vs_t.h5", 'a') as f:
        g0 = f.create_group("d0.0pi1")
        t0 = 1
        t1 = 10
        for (l0,m0),(l1,m1),name in zip(
               [(0,0), (0,0), (1,1), (1,1), (1,-1)],
               [(0,0), (1,1), (0,0), (1,1), (1,-1)],
               ['e2e', 'e2p', 'p2e', 'p2p', 's2s']
              ):
            g2 = g.create_group(name)
            k, t, G = Response_vs_k(t0, t1, l0, m0, l1, m1)
            g2.create_dataset('k', data=k, dtype=np.float)
            g2.create_dataset('t', data=t, dtype=np.float)
            g2.create_dataset('G', data=G, dtype=np.complex)
       
if sys.argv[1]=='2':
    with h5py.File("OfficialData/MISstar_Response_vs_k.h5", 'a') as f:
        g0 = f.create_group("d1.0pi1")
        for (t0,t1) in [(1,1.5),(1,2),(1,4),(1,8)]:
            g = g0.create_group("t{}-{}".format(t0,t1))
            for (l0,m0),(l1,m1),name in zip(
               [(0,0), (0,0), (1,1), (1,1), (1,-1)],
               [(0,0), (1,1), (0,0), (1,1), (1,-1)],
               ['e2e', 'e2p', 'p2e', 'p2p', 's2s']
              ):
                g2 = g.create_group(name)
                k, G = Response_vs_k(t0, t1, l0, m0, l1, m1)
                g2.create_dataset('k', data=k, dtype=np.float)
                g2.create_dataset('Gk', data=G, dtype=np.complex)

def Response_vs_t_k(tau0, tau1, Ls, Ms, Lr, Mr):

    ids = {(0,0):0,(1,1):1,(1,-1):2,(1,0):3}[(Ls, Ms)]
    idr = {(0,0):0,(1,1):1,(1,-1):2,(1,0):3}[(Lr, Mr)]
    ks = np.linspace(0, 10, 100)
    ts = np.linspace(tau0, tau1, 21)
    G = np.zeros([len(ts), len(ks)], dtype=np.complex)
    for it, t in enumerate(ts):
        print(tau0, tau1, name)
        for ik, k  in enumerate(ks):
            
            y0 = np.zeros(9, dtype=np.complex)
            y0[ids] = 1
            params = (t, k, 0, 0, 1, 1e-15, 1)
            d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), [t, tau1], args=params)[-1]
            d = d[:9]+1j*d[9:] 
            G[it,ik] = d[idr]
    return ks, ts, G



if sys.argv[1]=='3':

    with h5py.File("OfficialData/MISstar_Response_vs_t_k.h5", 'a') as f:
        g0 = f.create_group("d0d0pi1")
        for (t0,t1) in [(1,2),(1,4),(1,8)]:
            g = g0.create_group("t{}-{}".format(t0,t1))
            for (l0,m0),(l1,m1),name in zip(
               [(0,0), (0,0), (1,1), (1,1), (1,-1)],
               [(0,0), (1,1), (0,0), (1,1), (1,-1)],
               ['e2e', 'e2p', 'p2e', 'p2p', 's2s']
              ):
                g2 = g.create_group(name)
                k, t, G = Response_vs_t_k(t0, t1, l0, m0, l1, m1)
                g2.create_dataset('k', data=k, dtype=np.float)
                g2.create_dataset('t', data=t, dtype=np.float)
                g2.create_dataset('Gk', data=G, dtype=np.complex)



