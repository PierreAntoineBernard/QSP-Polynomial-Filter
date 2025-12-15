# QSP Polynomial Construction & Inversion

Small Python workspace to build a target Laurent polynomial from Chebyshev/C-D kernels, find a partner polynomial satisfying the SU(2) constraint $P(z)P(1/z) + Q(z)Q(1/z) = 1$ and extract QSP angles.

## Files
- [main.py](main.py) — entry point that constructs the target polynomial and extracts angles.

- [Christo_Darb_Ker.py](Christo_Darb_Ker.py) — Chebyshev and Christoffel-Darboux utilities

- [get_angle.py](get_angle.py) — Laurent polynomial class and QSP inversion routines

- [get_partner/get_partner_poly_real_no_renorm.py](get_partner/get_partner_poly_real_no_renorm.py) — numerical optimizer to find partner polynomial 

## Requirements
- Python 3.8+
- numpy, matplotlib, torch, torchaudio

## Run
Execute the main script: main.py

- Edit `renorm` in [main.py](main.py) to manually scale the target polynomial to satisfy |P(z)|< 1.
- Set `visualise = True` in [main.py](main.py) to plot results from the optimizer.
- Set `check_less_than_1 = True` in [main.py](main.py) to plot the polynomial over ]-1,1[ to check if |P(z)|< 1.




