# ================================================================
# ----------------------------- IMPORTS -----------------------------
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import math

import torch
from torch.fft import fft
from torchaudio.transforms import FFTConvolve

from Christo_Darb_Ker import Chebyshev, norms_for_chebyshev, Christoffel_Darboux_Kernel, eval_poly, convert_poly_to_laurent
from get_angle import *
from get_partner_poly_real_no_renorm import *


# ================================================================
#               Construction the target polynomial
# ================================================================
# Here, we construct the coefficients of the Laurent polynomial 
# corresponding to polynomial we want to implement via QSP.
# In particular, here we consider the Chebyshev Christoffel-Darboux kernel.
#----------------------------------------------------------------------------

y = 0 # point between -1 and 1 where we want to peak
N = 10  # degree of the polynomial, inversly proportional to width peak

# Matrix encoding the coefficients of Chebyshev polynomials up to degree N
Chebshev_matrix = Chebyshev(N) 

# Normalisation constants for Chebyshev polynomials
normalisation_Chebyshev = norms_for_chebyshev(N)

# Christoffel-Darboux kernel of Chebyshev, at point y
kernel = Christoffel_Darboux_Kernel(Chebshev_matrix, normalisation_Chebyshev, y)/N**2

# Conversion of polynomial in (z + z^{-1})/2 to Laurent polynomial in z
laurent_coeffs = convert_poly_to_laurent(kernel, N)


#Do you want to check if smaller than 1?
check_less_than_1 = False
if check_less_than_1:
    x = np.linspace(-1,1,500)
    ys = np.array([eval_poly(kernel, xi) for xi in x])
    plt.plot(x, ys)
    plt.title('Christoffel-Darboux Kernel at y=0, real line')  
    plt.show()

#===============================================================
#                Deriving the partner polynomial
#===============================================================
# Here, we construct the polynomial Q(z) partner to P(z) such that
# the su(2) constraint P(z)P(1/z) + Q(z)Q(1/z) = 1 is verified
# We do this via numerical optimisation, minimising the objective function
#----------------------------------------------------------------------------

renorm = 0.035 # Manual renormalization of target polynomial
laurent_coeffs = laurent_coeffs/renorm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
partner, profile, renormalization = partner_poly(laurent_coeffs)

#Do you want to see the result?
visualise = False

#Visulation success of the optimisation
if visualise:
    plot_two_poly(laurent_coeffs/np.sqrt(renormalization), partner/np.sqrt(renormalization))
    plt.plot(range(len(profile)),profile,'o-')
    plt.title('Renormalization =' + str(renormalization))
    plt.show()


#===============================================================
#                Extraction of the angles
#===============================================================
# Here, we use the target polynomial P(z) and partner polynomial Q(z)
# to extract the angles via the invserse QSP procedure
#----------------------------------------------------------------------------

P0 = Laurent_polynomials(laurent_coeffs, N)
Q0 = Laurent_polynomials(1j*partner, N)

P1 = Laurent_polynomials(laurent_coeffs, N)
Q1 = Laurent_polynomials(1j*partner, N)

sequence_of_angles_0 = []
sequence_of_angles_1 = []

for k in range(N):
    angle0 = angle_from_polynomials(P0,Q0, 0)
    angle1 = angle_from_polynomials(P1,Q1, 1)
    print('angles:',angle0 , angle1)
    print('iteration:', k)

    P0,Q0 = QSP_zbasis_inverse(angle0, P0,Q0)
    P1,Q1 = QSP_zbasis_inverse(angle1, P1,Q1)

    P0.check_degree(1e-10)
    Q0.check_degree(1e-10)
    P1.check_degree(1e-10)
    Q1.check_degree(1e-10)

    sequence_of_angles_0.append(angle0)
    sequence_of_angles_1.append(angle1)

    print('# ----------------------')

#print('Sequence of angles type 0:', sequence_of_angles_0)
#print('Sequence of angles type 1:', sequence_of_angles_1)
print('Error:', np.max( np.abs( np.array(sequence_of_angles_0) - np.array(sequence_of_angles_1) ) ) )

