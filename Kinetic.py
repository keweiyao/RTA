#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import lpmv, jv
from scipy.interpolate import interp1d, interpn
cm1, cm2 = plt.cm.Blues(.8), plt.cm.Reds(.8)
cb,co,cg,cr = plt.cm.Blues(.6), \
    plt.cm.Oranges(.6), plt.cm.Greens(.6), plt.cm.Reds(.6)
offblack = '#262626'
gray = '0.8'
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
    def __init__(self, getTauR, Lmax=3, N=11):
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
        for l in np.arange(4):
            self.Ylm[l] = {}
            self.Nlm[l] = {}
            for m in np.arange(-l,l+1):
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
            ylm = compute_Ylm(l,m,self.zeta, self.phi)
            nlm = (ylm**2).sum()*self.dOmega
            self.F += c*ylm/nlm + 0j

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

#################### some examples ########################################
# Complex initial conditon
def Example1():
    tau1 = 0.01
    t = 1
    fig, axes = plt.subplots(1,1,figsize=(4,4), sharex=True)
    for m, ax in enumerate([axes]):
        ax.set_title(r"$m={}, t={}$".format(m, t))
        #ax.plot([tau1, 10],[-(2*m+1)/2.,-(2*m+1)/2.], 'k--', label=r'$y=(2m+1)/2$')
        ax.annotate("[{}{}]/[{}{}]".format(m+t+2,m,m+t,m), xy=(.05,.9), xycoords='axes fraction')
        ax.set_xlim(tau1, 1e1)
        #ax.set_ylim(-3,3)
        ax.set_xlabel(r"$\tau/\tau_R$", fontsize=12)
        if m==0:
            ax.set_ylabel(r"$C_{m+t+2,m} / C_{0,0}$", fontsize=12)
    plt.tight_layout(True)
    axes.semilogx()
    #axes[0].semilogy()
    IC1 = [
1, -1, 0.520572, 0.00487705, 0.81878, -0.33697, 0.305316, -0.307405, \
0.342607, 0.244862, -0.419908, 0.744202, -0.0476228, 0.982722, \
0.587683, -0.258321, 0.104146, 0.63941, 0.616829, 0.269374, 0.486107, \
0.457193, -0.630011, -0.983191, -0.0726322, 0.132873, -0.101546]
    IC2 = [
1, -3, 0.935959, 0.0287309, -0.886935, 0.115042, 0.172002, \
-0.892546, -0.781497, -0.61862, 0.521252, -0.529452, -0.974002, \
0.874765, -0.865254, -0.614215, -0.689495, 0.0652648, -0.955551, \
0.404687, 0.865925, 0.947573, -0.819372, -0.897291, 0.901737]
    f = [linearRTA(getTauR=lambda x: 1, Lmax=len(2*IC1)) for i in range(2)]
    f[0].initialize(tau1, [(2*i+1,0,c) for i, c in enumerate(IC1)], kT=0, kappa=0)
    f[1].initialize(tau1, [(2*i+1,0,c) for i, c in enumerate(IC2)], kT=0, kappa=0)

    file = open("tau0={}.dat".format(tau1),'w')
    for j in range(3000):
        print(j)
        dtau = np.min([f[0].tau*.002,.002])
        for ff in f:
            ff.evolve(dtau=dtau, FS=False)
        for ax, m in zip([axes], [0]):      
            yy = []
            for ff, c, label in zip(f, ['g','orange'],
                                  [r"$L_{10}=L_{30}=\cdots L_{90}=1$", 
                                 r"$L_{10}=1/10, L_{30}=4/3$"]):
                A, B = ff.projection(m+t+2,m).real, ff.projection(m+t,m).real
                Y = A/B
                print(A, B)
                ax.plot(ff.tau, Y, '.', color=c, alpha=.6, ms=4, label=label if j==0 else'')
                yy.append(Y)
            file.write("{:1.3e}\t{:1.3e}\t{:1.3e}\n".format(ff.tau, *yy))
    ax.legend()
    plt.tight_layout(True)
    plt.savefig("plots/Ratio_t={}_tau0={}.png".format(t, tau1), dpi=300)
    plt.show()
import matplotlib.gridspec as gridspec
# power exponent of Clm
def Example2():
    tau1 = 10
    fig = plt.figure(constrained_layout=True, figsize=(7,7))

    gs = fig.add_gridspec(5,10)
    ax0 = fig.add_subplot(gs[0,-4])
    axes = [
      [ fig.add_subplot(gs[1, i:i+2]) for i in [4] ],
      [ fig.add_subplot(gs[2, i:i+2]) for i in [3,5] ],
      [ fig.add_subplot(gs[3, i:i+2]) for i in [2,4,6] ],
      [ fig.add_subplot(gs[4, i:i+2]) for i in [1,3,5,7] ],
      #[ fig.add_subplot(gs[4, i:i+2]) for i in [0,2,4,6,8] ],
    ]
    ni=5
    f = [linearRTA(getTauR=lambda x: 1, Lmax=7) for i in range(ni)]
    for i, aaa in enumerate([.1, .333, 1, 3, 10]):               
        f[i] = linearRTA(getTauR=lambda x: 1, Lmax=7)
        f[i].initialize(tau1, [2]+list(np.random.rand(8)), np.random.rand(9), kT=0, kappa=0, aaa=aaa)
    NL = 5
    C = {}
    for i in range(ni):
        C[i] = {}
        for l in range(NL): 
            C[i][l] = {}
            for m in range(-l, l+1):
                C[i][l][m] = []
    taus = []

    taus.append(f[0].tau)
    for i in range(ni):
        for l in range(NL): 
            for m in range(-l, l+1):
                C[i][l][m].append(f[i].projection(l,m))

    for j in range(1000):
        print(j)
        dtau = np.min([f[0].tau*.01, 0.02])
        taus.append(f[0].tau+dtau)
        for i in range(ni):
            f[i].evolve(dtau=dtau, FS=False)
            for l in range(NL): 
                for m in range(-l, l+1):
                    C[i][l][m].append(f[i].projection(l,m))
    taus = np.array(taus)
    
    
    for l in range(1,NL):
        for m in np.arange(-l,l+1):
            
            for i in range(ni):
                c = plt.cm.jet(i/4.01)
                y = np.array(C[i][l][m]).real
                ym = (y[1:]+y[:-1])/2.
                tm = (taus[1:]+taus[:-1])/2.
                dydt = (y[1:]-y[:-1]) / (taus[1:]-taus[:-1])
                
                if (l+m+1)%2==0:
                    print(l, m, (l+m)//2)
                    ax = axes[l-1][(m+l)//2]
 
                    ax.set_ylim(-4,1)
                    if i==0:
                        ref = tm/ym*dydt
                    EW = tm/ym*dydt #- ref
                    ax.plot(tm, EW, '-', color=c,lw=1)
                    #ax.semilogx()
                    ax.set_title(r"$lm={},{}$".format(l,m), fontsize=10)

                    ax.set_xlim(tau1, 20)
                    ax.set_xticks([tau1,20])
                    if ax.is_last_row():
                        ax.set_xlabel(r"$\tau/\tau_R$", fontsize=10)
                  
                #if ax.is_first_col():
                    #ax.set_ylabel(r"$d \ln C_{lm} /d\ln\tau$", fontsize=10)
    ax0.axis("off")
    ax0.annotate(r"$\frac{{d \ln C_{{lm}} }}{{d\ln\tau}}, \frac{{\tau_0}}{{\tau_R}}={:1.3f}$".format(tau1), xy=(.1,.5), xycoords="axes fraction", fontsize=16)
    plt.savefig("plots/Exponent_odd_tau0={}.png".format(tau1), dpi=300)
    #plt.show()


def background():
    tau1 = 0.1
    f = [linearRTA(getTauR=lambda x: 1, Lmax=2) for i in range(3)]
    for i, aaa in enumerate([.333, 1, 3]):               
        f[i] = linearRTA(getTauR=lambda x: 1, Lmax=2)
        f[i].initialize(tau1, [15,0,0,0,0,0,0,0,0], kT=0, kappa=0, aaa=aaa)
    e = []
    pi = []
    t = []
    T = tau1
    de = []
    dpi = []
    for i in range(3):
        L20 = f[i].projection(2,0).real # L-T
        L00 = f[i].projection(0,0).real # 2*T+L    L = (2*L20+L00)/3, T = (L00-L20)/3
        #Y = (2*L20+L00)/(L00-L20)
        de.append(L00)
        dpi.append(-L20/3.)
    e.append(de)
    pi.append(dpi)
    t.append(T)
    for j in range(1000):
        if j%50==0:
            print(j)
        dtau = np.min([f[0].tau*.01, 0.05])
        T += dtau
        de = []
        dpi = []



        for i in range(3):
            f[i].evolve(dtau=dtau, FS=False)
            L20 = f[i].projection(2,0).real # L-T
            L00 = f[i].projection(0,0).real # 2*T+L    L = (2*L20+L00)/3, T = (L00-L20)/3
            #Y = (2*L20+L00)/(L00-L20)
            de.append(L00)
            dpi.append(-L20/3.)
        e.append(de)
        pi.append(dpi)
        t.append(T)
    e = np.array(e)
    pi = np.array(pi)
    t = np.array(t)
    with open("Kinetic-Background.npy",'wb') as f:
        np.savez(f, e=e.T, pi=pi.T, t=t)


def Response_vs_k(tau0, tau1, Ls, Ms, Lr, Mr):
    ks = np.linspace(0,30,300)
    G = np.zeros_like(ks, dtype=np.complex)
    for ik, k in enumerate(ks):
        if ik % 10==0:
            print(tau0, tau1, name, ik)
        f = linearRTA(getTauR= lambda x: 1., Lmax=2)
        f.initialize(tau0=tau0, LMCs=[(Ls,Ms,1.0)], kT=k, kappa=0.)
        while f.tau < tau1:
             dtau = np.min([f.tau*.025, .025])
             f.evolve(dtau=dtau, FS=False)
        G[ik] = f.projection(Lr, Mr)
    return ks, G

def Response_vs_t_k(tau0, tau1, Ls, Ms, Lr, Mr):
    ks = np.linspace(0, 10, 100)
    ts = np.linspace(tau0, tau1, 21)
    G = np.zeros([len(ts), len(ks)], dtype=np.complex)
    for it, t in enumerate(ts):
        for ik, k  in enumerate(ks):
            if (ik%50==0):
                print(t, tau1, name, ik)
            f = linearRTA(getTauR= lambda x: 1., Lmax=2)
            f.initialize(tau0=t, LMCs=[(Ls,Ms,1.0)], kT=k, kappa=0.)
            while f.tau < tau1:
                dtau = np.min([f.tau*.05, .05])
                f.evolve(dtau=dtau, FS=False)
            G[it,ik] = f.projection(Lr, Mr)
    return ks, ts, np.array(G, dtype=np.complex)

def Response3D_vs_t_k(tau0, tau1, Ls, Ms, Lr, Mr):
    ks = np.linspace(0, 5, 30)
    kappas = np.linspace(0, 5, 30)
    ts = np.linspace(tau0, tau1, 11)
    G = np.zeros([len(ts), len(ks), len(kappas)], dtype=np.complex)
    for it, t in enumerate(ts):
        for ik, k  in enumerate(ks):
            for iK, K in enumerate(kappas):
                f = linearRTA(getTauR= lambda x: 1., Lmax=1)
                f.initialize(tau0=t, LMCs=[(Ls, Ms, 1.0)], kT=k, kappa=K)
                while f.tau < tau1:
                    dtau = np.min([f.tau*.05, .05])
                    f.evolve(dtau=dtau, FS=False)
                G[it,ik,iK] = f.projection(Lr, Mr)
    return ks, kappas, ts, np.array(G, dtype=np.complex)


from multiprocessing import Pool

# call the examples here
if __name__ == "__main__":
    #Example1()
    #all()
    import h5py
    
    """with h5py.File("OfficialData/Kinetic_Response_vs_k.h5", 'a') as f:
        for (t0,t1) in [(1,1.5),(1,2),(1,4),(1,8)]:
            g = f.create_group("t{}-{}".format(t0,t1))
            for (l0,m0),(l1,m1),name in zip(
               [(0,0), (0,0), (1,1), (1,1), (1,-1)],
               [(0,0), (1,1), (0,0), (1,1), (1,-1)],
               ['e2e', 'e2p', 'p2e', 'p2p', 's2s']
              ):
                g2 = g.create_group(name)
                k, G = Response_vs_k(t0, t1, l0, m0, l1, m1)
                g2.create_dataset('k', data=k, dtype=np.float)
                g2.create_dataset('Gk', data=G, dtype=np.complex)"""

    with h5py.File("OfficialData/Kinetic_3D_vs_t_k.h5", 'a') as f:
        for (t0,t1) in [(1,6)]:
            g = f.create_group("t{}-{}".format(t0,t1))
            for (l0,m0),(l1,m1),name in zip(
       [(0,0), (0,0), (0,0), (1,0), (1,0), (1,0), (1,1), (1,1), (1,1), (1,-1)],
       [(0,0), (1,0), (1,1), (0,0), (1,0), (1,1), (0,0), (1,0), (1,1), (1,-1)],
       ['e2e', 'e2l', 'e2p', 'l2e', 'l2l', 'l2p', 'p2e', 'p2l', 'p2p', 's2s']
              ):
                print(name)
                g2 = g.create_group(name)
                k, kappa, t, G = Response3D_vs_t_k(t0, t1, l0, m0, l1, m1)
                g2.create_dataset('t', data=t, dtype=np.float)
                g2.create_dataset('k', data=k, dtype=np.float)
                g2.create_dataset('kappa', data=kappa, dtype=np.float)
                g2.create_dataset('Gk', data=G, dtype=np.complex)
       
    


