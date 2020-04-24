# A numerical solver for the p-integrated RTA-Boltzmann equation describing small perturbations on a Bjorken flow.

# Download

```sh
$ git clone https://github.com/keweiyao/RTA.git
```

# Usage
`DiffInit.py` solves the equation and get the response functions $$A_{lm}(t)/A_{l_0m_0}(t_0)$$ where $$t=\tau/\tau_R$$. And $$A_{lm} = \int \frac{d\zeta d \phi}{4\pi}F(\tau, k_T, \kappa, \zeta, \phi) Y_{l_0m_0}(\zeta,\phi)\equiv \langle F Y_{lm}\rangle$$

```sh
$ ./DiffInte.py <label> <kT> <kappa> <l0> <m0>
```
  - `<kT>`: transverse wave-number [GeV]
  - `<kappa>`: longitudinal wave0number [Unity]
  - `<l0>` and `<m0>`: initialize the momentum space with $$Y_{l_0 m_0}$$ component.
  - `<label>`: results will be saved to `./Data/<label>/<l0><m0>_Re.dat` and `./Data/<label>/<l0><m0>_Im.dat` in a format `# ln10(t/tR), A00, A0-1, A00, A01, ...`

`make_plots.py` plots the calculated response function as function of $$\log_{10}(\tau/\tau_R)$$. One need to change the file reading path in the script to the one you intend to plot `./Data/<label>/<l0><m0>_Re.dat`. Then,

```sh
$ ./make_plots.py [plots/<function>]
```
It will generate plots in the folder `./plots/`

