#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.special import jv


tauR = 1.
GAMMAS = tauR/5.
cs2 = 1./3.
one_and_cs2 = 1. + cs2


def FreeTransport(tau, y, kx, ky, kappa):
    e, gx, gy, geta = y[:4]
    Fe = - one_and_cs2/tau*e - 1j*(kx*gx+ky*gy+kappa*geta/tau)
    Fgx = - gx/tau - cs2*1j*kx*e 
    Fgy = - gy/tau - cs2*1j*ky*e 
    Fgeta = -2.*geta/tau-cs2*1j*kappa/tau*e
    return np.array([Fe, Fgx, Fgy, Fgeta])
    
def NS_dissipation(tau, y, kx, ky, kappa, gammas):
    e, gx, gy, geta = y[:4]
    De =  0 
    Dgx = -gammas*(
              (one_and_cs2*kx**2+ky**2+(kappa/tau)**2)*gx
              + cs2*kx*ky*gy + cs2*kx*kappa*geta/tau 
              )
    Dgy = -gammas*(
          cs2*kx*ky*gx\
        + (kx**2+one_and_cs2*ky**2+(kappa/tau)**2)*gy\
        + cs2*ky*kappa*geta/tau
       )
    Dgeta = -gammas*(
             cs2/tau*kappa*(kx*gx+ky*gy)
          + (kx**2+ky**2+one_and_cs2*(kappa/tau)**2)*geta
        )
    return np.array([De, Dgx, Dgy, Dgeta])

def IS_disspation(tau, y, kx, ky, kappa, gammas):
    taupi = tauR/1.3
    tauCR = tauR
    e, gx, gy, geta, L200, L210, L211, L220, L221 = y
    De = 0
    Dgx = cs2*1j*kx*L200  \
              - 1j*kappa/tau/3.*L210\
              - 1j/6.*(kx*L220+ky*L221)
    Dgy = cs2*1j*ky*L200  \
              - 1j*kappa/tau/3.*L211\
              - 1j/6.*(-ky*L220+kx*L221)
    Dgeta = -cs2*1j*kappa/tau*2.*L200\
                -1j/3.*(kx*L210+ky*L211)

    ## >>> set L3 terms to zero to go back to IS theory
    ## use the graident term to go to 2nd order theory
    L300 = 1j*tauCR*(kx*L210/7+ky*L211/7-kappa/tau*L200*3/7)
    L310 = 1j*tauCR*(-kx*L200*6/7+kx*L220/14+ky*L221/14-kappa/tau*L210*4/7)
    L311 = 1j*tauCR*(-ky*L200*6/7+kx*L221/14-ky*L220/14-kappa/tau*L211*4/7)
    L320 = 1j*tauCR*(-kx*L210*10/7+ky*10/7*L211-kappa/tau*L220*5/7)
    L321 = 1j*tauCR*(-kx*L211*10/7-ky*10/7*L210-kappa/tau*L221*5/7)
    L330 = 1j*tauCR*(-kx*15/7*L220+ky*L221*15/7)
    L331 = 1j*tauCR*(-kx*15/7*L221-ky*L220*15/7)
    ### <<<

    NS200 = 1j*gammas*(kx*gx+ky*gy+4./21.*e/tau-2.*kappa/tau*geta
             #-kx*L310 - ky*L311 - 3.*kappa/tau*L300
             )
    NS210 = -1j*gammas*3.*(kx*geta+kappa/tau*gx
             #         -kx*L300 + kx/6.*L320
             #         +ky/6.*L321 + kappa/tau*2./3.*L310
            ) 
    NS211 = -1j*gammas*3.*(ky*geta+kappa/tau*gy
             #         -ky*L300 - ky/6.*L320\
             #         +kx/6.*L321 + kappa/tau*2./3.*L311
            )
    NS220 = -1j*gammas*6.*(kx*gx-ky*gy
             #         -kx*L310/6.+kx*L330/12.\
             #         +ky*L311/6. + ky*L331/12.
             #        +kappa/tau/6.*L320
                )
    NS221 = -1j*gammas*6.*(ky*gx+kx*gy
             #         -ky*L310/6.-ky*L330/12.\
             #         -kx*L311/6. + kx*L331/12.
             #        +kappa/tau/6.*L321
            )
    D200 = - (L200-NS200)/taupi - 38./21./tau*L200
    D210 = - (L210-NS210)/taupi - 33./21./tau*L210
    D211 = - (L211-NS211)/taupi - 33./21./tau*L211
    D220 = - (L220-NS220)/taupi - 18./21./tau*L220
    D221 = - (L221-NS221)/taupi - 18./21./tau*L221
    return np.array([De,Dgx,Dgy,Dgeta,D200,D210,D211,D220,D221])

def IS_extended(ay, tau, kx, ky, kappa, gammas1, gammas2):
    y = ay[:9] + 1j*ay[9:]
    Dy = np.zeros(9, dtype=np.complex)
    Dy[:4] += FreeTransport(tau, y, kx, ky, kappa)
    Dy[:4] += NS_dissipation(tau, y, kx, ky, kappa, gammas1)
    Dy += IS_disspation(tau, y, kx, ky, kappa, gammas2)
    return np.concatenate([Dy.real, Dy.imag])
    
class hydro:
    def __init__(self, g):
        self.y0 = None
        self.tauR=1.0
        self.g=g
        self.tau1=None
        self.tau2=None
        self.karray=None
        self.res=None
    def initialize(self, l0, m0): # g=1: NS, g=0: IS
        self.l0=l0
        self.m0=m0
        self.y0 = np.zeros(9)
        if l0==0 and m0==0:
            self.y0[0]=1
        if l0==1 and m0==1:
            self.y0[1]=1
        if l0==1 and m0==-1:
            self.y0[2]=1
    def solve(self, tau1, tau2):
        self.tau1 = tau1
        self.tau2 = tau2
        self.karray = np.linspace(0, 50/self.tauR, 200)/(self.tau2-self.tau1)
        self.res = []
        for k in self.karray:
            solver = ode(IS_extended).set_integrator('zvode', method='bdf')
            solver.set_f_params(kT, 0., 0., 
                   GAMMAS*self.g, GAMMAS*(1-self.g)
                 ).set_initial_value(self.y0, self.tau1)
            self.res.append(solver.integrate(self.tau2)[:3])
        self.res = np.array(self.res).T
        for idx, (l, m) in enumerate([(0,0),(1,1),(1,-1)]):
            fname = "data/Response_at_fixed_t1_t2/Hydro/g=0d2-taupi1d5-IS/{}-to-{}-{}{}-to-{}{}.npy".format(self.tau1, self.tau2, 
                 self.l0, self.m0, l, m)
            np.savez(fname, k=self.karray, RTA=self.res[idx])


def Gk(g,t1,t2, l0, m0):
    solver = hydro(g) 
    solver.initialize(l0, m0)
    solver.solve(t1,t2)


def Tabulate(g=0.12):
    taui = 1
    tauf = 11
    kTarray = np.linspace(0,10,101)
    tarray = np.linspace(taui, tauf, 101)
    dt = tarray[1]-tarray[0]
    res = {'kT':kTarray, 't1':tarray, 't2':tarray}
    for (l0, m0, l, m), name in zip([(0,0,0,0),(1,1,1,1),
                      (1,-1,1,-1),(1,1,0,0),(0,0,1,1)],
                        ['ee','pp','tt','pe','ep']):
        y0 = np.zeros(18)
        if l0==0 and m0==0:
            y0[0]=1
        if l0==1 and m0==1:
            y0[1]=1
        if l0==1 and m0==-1:
            y0[2]=1
        if l==0 and m==0:
            idx=0
        if l==1 and m==1:
            idx=1
        if l==1 and m==-1:
            idx=2
        Response = np.zeros([len(kTarray), len(tarray), len(tarray)],
                             dtype=np.complex)
        for k, kT in enumerate(kTarray):
            print("kT = ", kT)
            for i, t1 in enumerate(tarray):
                ay = odeint(IS_extended, y0, tarray[i:],
                  args=(kT, 0., 0., GAMMAS*g, GAMMAS*(1-g))
                )
                Response[k,i,i:]  = ay[:,idx] + 1j*ay[:,9+idx]

        res[name] = Response
    np.savez('Extend_0d12_1d3.npz', **res)

if __name__ == "__main__":
    Tabulate()


