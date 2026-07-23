# dynamics-FP
Code and data for "Non-equilibrium Dynamics of Fermi Polarons Driven by Time-dependent Interactions"

Numerically calculate the nonequilibrium dynamics of a Fermi polaron driven by time-dependent interactions. 数值计算时变相互作用驱动下费米极化子的非平衡动力学。

## Usage

Place the Python scripts, Jupyter notebooks, and `fermi_polaron_trapezoid_optimized.py` in the same directory.

Run the programs as follows:

- Run `fermi_polaron_figure2.py` to reproduce Fig. 2.
- Run `fermi_polaron_figure3.py` to reproduce Fig. 3.
- Run `fermi_polaron_figure4_compute_logk_v3.ipynb` in Jupyter to calculate the momentum-distribution data for Fig. 4.
- Run `fermi_polaron_figure4_plot_combined_logk_v3.ipynb` in Jupyter to read the calculated data and plot Fig. 4.

For Fig. 4, run the compute notebook before the plot notebook.

## Program Overview

### 1. Optimized Solver

`fermi_polaron_trapezoid_optimized.py` solves the time-dependent integral equation for \(\chi_q(t)\) using the trapezoidal/Hadamard method.

The program calculates:

- Quasiparticle residue
- Contact
- Energy
- Impurity momentum distribution

### 2. Figure 2

Calculates the contact and quasiparticle residue for different interaction-ramp strengths.

### 3. Figure 3

Calculates the energy dynamics for different interaction-ramp strengths.

### 4. Figure 4

Calculates the impurity momentum distribution \(n_k\).

The momentum grid is uniformly spaced in \(\log k\). The oscillatory time integral is evaluated using a piecewise-linear Fourier transform and an oversampled fast Fourier transform.

## Main Parameters

The main numerical parameters can be changed directly in the notebooks:

- `N_Q`: number of momentum-grid intervals
- `DT`: time step
- `N_T`: number of time steps
- `N_SEED`: number of short-time seed points
- `N_K_POINTS`: number of momentum points for Fig. 4
- `FFT_OVERSAMPLING`: fast Fourier transform oversampling factor

The numerical results should be checked for convergence with respect to these parameters.

## Requirements

- Python
- NumPy
- SciPy
- Matplotlib
- tqdm
- Jupyter Notebook

## Data

The Fig. 4 compute notebook generates:

- `figure4_beta_1_nk.npz`
- `figure4_beta_0p2_nk.npz`

These files are read by the Fig. 4 plot notebook.

## Reference

These programs accompany the manuscript:

*Non-equilibrium Dynamics of Fermi Polarons Driven by Time-dependent Interactions*
