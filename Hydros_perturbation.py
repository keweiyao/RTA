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
power = 0

def GetTauR(tau0, tau):
    return tauR0 * (tau/tau0)**power


def Background(tau, y, tau0, l1):
    E, Pi = y[:2]
    taupi = tauR = GetTauR(tau0, tau)
    Pi_eq = tauR*8/(45*tau)*E
    dEdtau = - 4.*E/(3.*tau) + 2*Pi/tau
    dPidtau = - (Pi-Pi_eq)/taupi \
              - Pi/tau*(4./3.+2./3.*lambda1*l1)
    return [dEdtau, dPidtau]

def compute_pi(tau, y, tau0, kx, ky, kappa, l1, g, piscale):
    tauR = GetTauR(tau0, tau)
    gammas = tauR/5.
    cg = 1.-g
    taupi = tauR*piscale
    E, Pi, e, gx, gy, geta, L200, L210, L211, L220, L221 = y
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
    return g*pixxeq+pixx, g*pixyeq+pixy, g*pixzeq+pixz, \
           g*piyyeq+piyy, g*piyzeq+piyz, g*pizzeq+pizz, \
           D200, D210, D211, D220, D221

def Perturbation_consrvation_equation(tau, y, tau0, kx, ky, kappa, l1, gscale, piscale):
    E, Pi, e, gx, gy, geta, L200, L210, L211, L220, L221 = y
    P = E/3.
    H = (E+P)
    pixx, pixy, pixz, piyy, piyz, pizz,\
    D200, D210, D211, D220, D221 = compute_pi(tau, y, tau0, kx, ky, kappa, l1, gscale, piscale)
    dEdtau, dPidtau = Background(tau, y, tau0, l1)
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
    y = Y[:11] + 1j*Y[11:]
    Dy = np.zeros(11, dtype=np.complex)
    Dy[:2]  += Background(tau, y, tau0, l1)
    Dy[2:]  += Perturbation_consrvation_equation(tau, y, tau0, kx, ky, kappa, l1, delta_plus, piscale)
    return np.concatenate([Dy.real, Dy.imag])


def plot_background_solution(t0=1e-1, t1=1e2):
    fig, ax = plt.subplots(1,1,figsize=(5,4), sharex=True)
    e0 = 15

    ts = np.exp(np.linspace(np.log(t0), np.log(t1), 1000))
    
  
    y0 = np.array([e0,0,e0,0,0,0, 0,0,0,0,0])
    
    for pioverh, line,file in zip([0,0.8],['-','-.','--'],
                       ['pi0.dat','pi0d8.dat']): 
        for lrescale, color, label in zip([1.0, 7/5], [plt.cm.Reds(.5), plt.cm.Blues(.5), plt.cm.Greens(.5)],
           ['$\lambda_1=5/7$, DNMR', '$\lambda_1=1$, Free-stream']):
            params = (t0, 0, 0, 0, lrescale, 0, 1)
            y0[1] = pioverh*e0*0.75
            y0[6] = -3*y0[1]
            res = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params).T
            res = res[:11]+1j*res[11:] 
            A = 0.75*res[6]*(-1./3.)/res[2]
            ax.plot(ts/GetTauR(t0, ts), A, linestyle=line, color=color, label=label if line=='-' else'',lw=2)

        file = "RTA_background/"+file

        x,y = np.loadtxt(file).T
        ax.plot(x/GetTauR(t0, x),y,line,color='k',label='Kinetic'if line=='-' else'',lw=2)
    ax.semilogx()
    #ax.semilogy() 
    ax.legend()
    #ax.set_ylim(2e-2,0.5)
    ax.set_xlabel(r"$\tau/\tau_R(\tau)$",fontsize=15)
    ax.set_ylabel(r"$e\tau^{4/3} / e_f \tau_f^{4/3}$",fontsize=15)
    ax.set_ylabel(r"$\pi/(e+P)$",fontsize=15)
    plt.tight_layout(True)
    plt.subplots_adjust(top=.9)
    plt.savefig("Background_Solution.png", dpi=300)
    plt.show()

def plot_Gk_tau(t0=1e-1, t1=1e1):
    fig, ax = plt.subplots(1,1,figsize=(5,4), sharex=True)
    e0 = 15

    ts = np.exp(np.linspace(np.log(t0), np.log(t1), 10000))
    
  
    y0 = np.array([e0, 0,
                   0, 
                   1, 0, 0, 
                   0, 0, 0, 0, 0])
    
    for ik, k in enumerate([0, 1, 2, 4]):      
        params = (t0, k, 0, 0, 7/5, 1e-8, 1)
        res = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params).T
        res = res[:11]+1j*res[11:] 
        color = plt.cm.coolwarm(ik/3)
        ax.plot(ts/GetTauR(t0, ts), np.abs(res[2]), color=color, label=r"$k\tau_{{R,0}}={}$".format(k), lw=2)
        
    ax.semilogx()
    ax.semilogy() 
    ax.legend()
    #ax.set_ylim(2e-2,0.5)
    ax.set_xlabel(r"$\tau/\tau_R(\tau)$",fontsize=15)
    ax.set_ylabel(r"$|G_{v}^{v}(\tau', \tau)|$",fontsize=15)
    plt.tight_layout(True)
    plt.subplots_adjust(top=.9)
    plt.savefig("Gk_pe_tau.png", dpi=300)
    plt.show()


def plot_Gk_k(t0=1, t1=5):
    fig, ax = plt.subplots(1,1,figsize=(5,4), sharex=True)
    e0 = 15

    ts = np.exp(np.linspace(np.log(t0), np.log(t1), 100))
    
  
    y0 = np.array([e0, 0,
                   1, 
                   0, 0, 0, 
                   0, 0, 0, 0, 0])
    ks = np.linspace(0,10,100)

    x, y = np.loadtxt("KineticResponse/Gs_k_tau1_5.dat".format(t0,t1)).T
    ax.plot(x,y*(t1/t0)*1.3333,'k-', label='Kinetic')

    

    res = []
    for k in ks:
        params = (t0, k, 0, 0, 1, 1-1e-9, 1)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params)[-1]
        d = d[:11]+1j*d[11:] 
        res.append(d[0]/15)
    res = np.array(res)

    ax.plot(ks, res*(t1/t0)**1.3333, 'k--', label='NS')
  
    res = []
    for k in ks:
        params = (t0, k, 0, 0, 1, 1e-9, 1)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params)[-1]
        d = d[:11]+1j*d[11:] 
        res.append(d[2])
    res = np.array(res)

    ax.plot(ks, res*(t1/t0)**1.3333, color=plt.cm.Reds(.5), lw=2, 
            label=r'IS, $\lambda_1=5/7, \delta=0,\tau_\pi=\tau_R$')

    res = []
    for k in ks:
        params = (t0, k, 0, 0, 1, 0.12, .5)
        d = odeint(MIS_star, np.concatenate([y0.real, y0.imag]), ts, args=params)[-1]
        d = d[:11]+1j*d[11:] 
        res.append(d[2])
    res = np.array(res)

    ax.plot(ks, res*(t1/t0)**1.3333, color=plt.cm.Blues(.5), lw=2, 
            label=r'IS, $\lambda_1=5/7, \delta=0,\tau_\pi=\tau_R$')
 
    ax.legend()
    ax.plot([0,10],[0,0],'k-')
    ax.set_xlabel(r"$k_s$",fontsize=15)
    ax.set_ylabel(r"$G_s(\tau', \tau; k)$",fontsize=15)
    plt.tight_layout(True)
    plt.subplots_adjust(top=.9)
    plt.savefig("Gk_k.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    #plot_background_solution()
    #plot_Gk_tau()
    plot_Gk_k()

