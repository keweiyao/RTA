#!/usr/bin/env python3
#import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lpmv, jv
from scipy.interpolate import interp1d, interpn
from multiprocessing import Pool
from itertools import repeat

# compute Ylm(xi, yj) 
def compute_Ylm(l,m,x,y):
    if m==0:
        return np.outer(lpmv(m, l, x), np.ones_like(y))
    if m>0:
        return (-1)**m*np.outer(lpmv(m, l, x), np.cos(m*y))
    if m<0:
        return (-1)**(-m)*np.outer(lpmv(-m, l, x), np.sin(-m*y))

# The linearized RTA class
class linearRTA:
    def __init__(self, getTauR, Lmax=3, N=101):
        self.TauR = getTauR 
        self.tau0 = None
        # Discretize the angular space of momentum: 
        #    Omega = (zeta=cos(theta), phi)
        self.N = N
        # -1 < zeta < 1
        self.zeta = np.linspace(-1., 1., N+1)
        self.zeta = (self.zeta[:-1]+self.zeta[1:])/2.
        self.dzeta = self.zeta[1]-self.zeta[0]
        # zetaT = sqrt(1-zeta^2)
        self.zetaT = np.sqrt(1.-self.zeta**2)
        # -pi < phi < pi 
        self.phi = np.linspace(-np.pi, np.pi, N+1)
        self.phi = (self.phi[:-1]+self.phi[1:])/2.
        self.cosphi = np.cos(self.phi)
        self.dphi = self.phi[1]-self.phi[0]
        # The phase-space average factor dOmega = dx*dy/4/Pi
        self.dOmega = self.dzeta*self.dphi/(4.*np.pi)

        # pre-calculate a table for Ylm to be used over and over again
        self.Ylm = {}
        self.Nlm = {}
        for l in np.arange(Lmax+1):
            self.Ylm[l] = {}
            self.Nlm[l] = {}
            for m in np.arange(-l, l+1):
                self.Ylm[l][m] = compute_Ylm(l,m,self.zeta, self.phi)
                self.Nlm[l][m] = (self.Ylm[l][m]**2).sum()*self.dOmega
        self.kT = None
        self.kappa = None
        self.F = np.zeros([self.N, self.N], dtype=np.complex)
        self.FI = np.zeros([self.N, self.N], dtype=np.complex)
        
    def initialize(self, tau0, LMCs, kT, kappa):
        self.tau0 = tau0
        self.tau = self.tau0
        self.tauR = self.TauR(self.tau0)
        self.kT = kT
        self.kappa = kappa
        # .....................

        for (l, m, c) in LMCs:
            self.F += self.Ylm[l][m]*c/self.Nlm[l][m] + 0j

        # store F in its interaction picture, at t=t0, they are the same
        self.FI = self.F

    def evolve(self, dtau, FS=False):
        self.tauR = self.TauR(self.tau)
        # define some useful varaibles
        t = self.tau/self.tau0
        a = np.sqrt((self.zetaT/t)**2 + self.zeta**2)
        # Phase factor for the interaction picture F = FI*U
        # U = exp(-i*Sk - i*Skappa)
        # cancels the k-dependent term in the equation
        Sk = self.kT*self.tau*np.outer((1.-a)/self.zetaT, self.cosphi)
        Skappa = self.kappa*np.outer(
                           .5*np.log( (1.+self.zeta/a)*(1.-self.zeta)  \
                                     /(1.-self.zeta/a)/(1.+self.zeta)  ),
                            np.ones_like(self.phi)
                           )
        U = np.exp(-1j*(Sk+Skappa))
        
        # FI staisfies equation:
        # dFI/dtau - zeta*zetaT^2/tau * dFI/dzeta \
        #       = - (FI*Ya*U^{-1}<FI*U*Ya>)/tauR
        # Discretize:
        # FI[n+1,i,j] = 
        #    FI[n,i,j]*(1-dtau/tauR) \
        #  + (zeta*zetaT^2)[i]/tau[n]/(2*dzeta)*(FI[n,i+1,j]-FI[n,i-1,j])
        #  + sum_{a} (Ya*U^{-1})[n,i,j] * sum_{kl} (F*Ya)[n,i,j]
        diag = 1. - (1/self.tauR + 4.*self.zeta**2/self.tau)*dtau
        offdiag = self.zeta*self.zetaT**2/self.tau*dtau/(2*self.dzeta)

        # Four conserved elements of F
        L000 = (self.Ylm[0][0]*self.F).sum()*self.dOmega
        L100 = (self.Ylm[1][0]*self.F).sum()*self.dOmega*3.
        L110 = (self.Ylm[1][1]*self.F).sum()*self.dOmega*3.
        L111 = (self.Ylm[1][-1]*self.F).sum()*self.dOmega*3.
        # Update FI
        FI_new = (self.FI.T*diag).T
        FI_new[:-1] += (self.FI[1:].T*offdiag[:-1]).T
        FI_new[1:] -= (self.FI[:-1].T*offdiag[1:]).T
        FI_new += dtau/self.tauR*np.conjugate(U)*( 
                      L000*self.Ylm[0][0] + L100*self.Ylm[1][0]
                    + L110*self.Ylm[1][1] + L111*self.Ylm[1][-1]
                    )
        # set the boundary of FI at zeta=-1 and 1
        # so that d^2F/dzeta^2 = 0, therefore dF/dzeta does not change
        # when approaching the boundary
        #FI_new[0] = 2*FI_new[1]-FI_new[2]
        #FI_new[-1] = 2*FI_new[-2]-FI_new[-3]
        # Store updated FI and F
        self.FI = FI_new
        self.F = self.FI*U
        # Move forward in time
        self.tau += dtau
        
    # project out the coefficients in the expansion:
    #      Clm = <Ylm*F>
    def projection(self, l, m):
        return (self.F*self.Ylm[l][m]).sum()*self.dOmega

def Response3D_vs_t_k(params):
    tau0, tau1, Ls, Ms, Lr, Mr, name = params
    ks = np.linspace(0, 10, 100)
    kappas = np.linspace(0, 10, 100)
    ts = np.linspace(tau0, tau1, 11)
    G = np.zeros([len(ts), len(ks), len(kappas)], dtype=np.complex)
    for it, t in enumerate(ts):
        for ik, k  in enumerate(ks):

            if (ik%100==0):
                print(t, tau1, name, ik)
            for iK, K in enumerate(kappas):
                f = linearRTA(getTauR= lambda x: 1., Lmax=1)
                f.initialize(tau0=t, LMCs=[(Ls, Ms, 1.0)], kT=k, kappa=K)
                while f.tau < tau1:
                    dtau = np.min([f.tau*.05, .05])
                    f.evolve(dtau=dtau, FS=False)
                G[it,ik,iK] = f.projection(Lr, Mr)
    return (ks, kappas, ts, np.array(G, dtype=np.complex))



LM0 = [(0,0), (0,0), (0,0), (1,0), (1,0), (1,0), (1,1), (1,1), (1,1), (1,-1)]
LM1 = [(0,0), (1,0), (1,1), (0,0), (1,0), (1,1), (0,0), (1,0), (1,1), (1,-1)]
channels = ['e2e', 'e2l', 'e2p', 'l2e', 'l2l', 'l2p', 'p2e', 'p2l', 'p2p', 's2s']

import h5py
t0, t1 = 1, 2
f = h5py.File("OfficialData/Kinetic_3D_vs_t_k.h5", 'a')
params = [(t0, t1, *a, *b, c) for a, b, c in zip(LM0, LM1, channels)]
with Pool(10) as p:
    res = p.map(Response3D_vs_t_k, params)
    
    g = f.create_group("t{}-{}".format(t0,t1))
    for i, name in enumerate(channels):
        g2 = g.create_group(name)
        k, kappa, t, G = res[i]
        g2.create_dataset('t', data=t, dtype=np.float)
        g2.create_dataset('k', data=k, dtype=np.float)
        g2.create_dataset('kappa', data=kappa, dtype=np.float)
        g2.create_dataset('Gk', data=G, dtype=np.complex)
 
