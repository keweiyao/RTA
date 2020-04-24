#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
from scipy.special import lpmv 
import sys, os

# define a plotting function, not really necessary
def plot_f_vs_costheta(f, fig, t, i):
    plt.clf()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    H = ax.plot_surface(X.T, Y.T, f.real, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.plot_surface(X.T, Y.T, f.imag-5, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel(r"$x=\cos\theta$")
    ax.set_ylabel(r"$y=\phi$")
    ax.set_zlabel(r"$\mathrm{Re}F(k)/T^4$ (upper), $\mathrm{Im}F(k)/T^4$ (lower)")
    ax.set_zlim(-7,2)
    plt.tight_layout(True)
    plt.title(r"$\tau/\tau_R = {:1.3f}$".format(t))
    plt.subplots_adjust(top=.9)
    plt.colorbar(H)
    #plt.savefig("plots/{:d}.png".format(i))
    plt.pause(.05)

if len(sys.argv) != 6:
    print(
"""
Usage
./DiffInte <label> <kT> <kappa> <l0> <m0>
"""
)
    exit(-1)
label = sys.argv[1]
kT = float(sys.argv[2])
kappa = float(sys.argv[3])
l0=int(sys.argv[4])
m0=int(sys.argv[5])
if l0<np.abs(m0):
    print("wrong l, m! -l<=m<=l")
    exit(-1)
 
# effective eta/s
etas = .2
# Initial time
tau0 = 0.02 * 5.076 # 1/GeV
# Relaxization time: ~ 1/Teff
def getTauR(tau):
    # Assume Teff(tau=0.2fm/c) = 1 GeV
    T0 = 1 # GeV
    return 5.*etas/T0*np.power(tau/(0.2*5.026),1./3.)

# Discretize F(tau, x=cos(theta), y=phi) into n by n (x, y) grids
N = 101
# x values distributed from -1+dx/2. ... to ... 1-dx/2.
x = np.linspace(-1., 1., N+1)
x = (x[1:]+x[:-1])/2.
dx = x[1]-x[0]
# xT = sqrt(1-x^2) = sin(theta)
xT = np.sqrt(1.-x**2)

# y values distributed from -pi+dy/2. ... to ... pi-dy/2.
y = np.linspace(-np.pi, np.pi, N+1)
y = (y[1:]+y[:-1])/2.
dy = y[1]-y[0]

# The phase-space average factor dOmega = dx*dy/4/Pi
dOmega = dx*dy/4./np.pi

# Intreaction picture phase coefficients
A = -1j*kT*np.outer(1./xT, np.cos(y))
B = 1j*kappa*np.outer(0.5*np.log((1+x)/(1.-x)), np.ones_like(y))

# compute Ylm(xi, yj) 
def compute_Ylm(l,m,x,y):
    if m==0:
        return np.outer(lpmv(m, l, x), np.ones_like(y))
    if m>0:
        return (-1)**m*np.outer(lpmv(m, l, x), np.cos(m*y))
    if m<0:
        return (-1)**(-m)*np.outer(lpmv(-m, l, x), np.sin(-m*y))

# pre-calculate a table for Ylm to be used over and over again
Ylm = {}
Lmax = 6 # max l index
for l in np.arange(Lmax+1):
    Ylm[l] = {}
    for m in np.arange(-l, l+1):
        Ylm[l][m] = compute_Ylm(l,m,x,y)

# set initial time
tau = tau0
lntau = np.log(tau/tau0)
# Initialize the state with a pure l0, m0 component
F = compute_Ylm(l0,m0,x,y)
# normalize F
F = F / ((F*F).sum()*dOmega)
# Convert F into the interaction picture F = FI*exp(S), FI = F*exp(-S)
S = tau*A+B
FI = F * np.exp(-S)
# we are going to record the time evolution of Alm = <F*Ylm>
Alm = {}
ln10t_tr = []
for l in np.arange(Lmax+1):
    Alm[l] = {}
    for m in np.arange(-l, l+1):
        Alm[l][m] = []
#fig = plt.figure()
for i in range(6000): 
    # compute tau from ln(tau/tau0)
    tau = tau0*np.exp(lntau)
    tauR = getTauR(tau)
    # set dln(tau/tau0) << tauR/tau, dx/tau, 1
    dlntau = .15*np.min([tauR/tau, 1, dx/tau])
    # diagonal elements array
    diag = 1. - (tau/tauR + 4*x**2)*dlntau
    # upper off-diagonal elements array
    offdiag = x*(1-x**2)*dlntau/(2*dx)
    S = tau*A+B
    phase = np.exp(S)
    cphase = np.conjugate(phase)
    F = FI * phase

    # compute Alm at x=y=eta=0 for this (k, kappa) mode
    if i%10==0:
        for l in np.arange(Lmax+1):
            for m in np.arange(-l,l+1):
                c = (F*Ylm[l][m]).sum()*dOmega
                Alm[l][m].append(c)
        ln10t_tr.append(np.log10(np.exp(lntau)*tau0/tauR))

    # uncomment for real-time plotting
    #if i%30 == 0:
    #    plot_f_vs_costheta(F, fig, tau/tauR, i)
    
    # Four conserved elements of F
    L000 = (Ylm[0][0]*F).sum()*dOmega
    L100 = (Ylm[1][0]*F).sum()*dOmega
    L110 = (Ylm[1][1]*F).sum()*dOmega
    L111 = (Ylm[1][-1]*F).sum()*dOmega
    # Update to lntau+dlntau
    FI_new = np.zeros([N, N], dtype=np.complex)
    FI_new += (FI.T*diag).T
    FI_new[:-1] += (FI[1:].T*offdiag[:-1]).T
    FI_new[1:] -= (FI[:-1].T*offdiag[1:]).T
    FI_new += tau/tauR*dlntau*cphase\
             *(L000*Ylm[0][0]+L100*Ylm[1][0]+L110*Ylm[1][1]+L111*Ylm[1][-1])
    FI = FI_new
    lntau += dlntau

# write real part
os.makedirs("Data/{:s}/".format(label), exist_ok=True)
with open("Data/{:s}/{:d}{:d}_Re.dat".format(label, l0, m0), 'w') as fre,\
     open("Data/{:s}/{:d}{:d}_Im.dat".format(label, l0, m0), 'w') as fim:
    fre.write("# log10(t/tR)\t")  
    fim.write("# log10(t/tR)\t")  
    for l in np.arange(Lmax+1):
        for m in np.arange(-l,l+1):
            fre.write("{:d},{:d}\t".format(l,m))  
            fim.write("{:d},{:d}\t".format(l,m))  
    fre.write("\n")  
    fim.write("\n")  
    for i, it in enumerate(ln10t_tr):
        fre.write("{:1.3e}\t".format(it))  
        fim.write("{:1.3e}\t".format(it))  
        for l in np.arange(Lmax+1):
            for m in np.arange(-l,l+1):
                fre.write("{:1.3e}\t".format(Alm[l][m][i].real))  
                fim.write("{:1.3e}\t".format(Alm[l][m][i].imag))    
        fre.write("\n") 
        fim.write("\n")  


