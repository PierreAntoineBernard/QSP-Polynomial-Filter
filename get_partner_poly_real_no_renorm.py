import matplotlib.pyplot as plt
import numpy as np
#from itertools import combinations
#import math

import torch
from torch.fft import fft
from torchaudio.transforms import FFTConvolve


np.random.seed(0)
torch.manual_seed(0)

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# CONVOLUTION (gradient-safe)
# -----------------------------
def convol_poly(poly):
    """
    Compute convolution of poly with its conjugate-reversed version.
    Works for complex torch tensors by splitting real and imaginary parts.

    poly: torch tensor (complex, requires_grad allowed)
    returns: torch tensor (complex)
    """
    # Original tensor
    coeff = poly
    # Conjugate and reversed
    coeff_rev = torch.flip(coeff, dims=[0])
    
    
    # Convolve real and imaginary parts separately
    conv_real = FFTConvolve("full").forward(coeff, coeff_rev) 
    
    # Recombine
    return conv_real 


# -----------------------------
# OBJECTIVE FUNCTION
# -----------------------------
def objective_function(x, target_conv, ratio):
    """
    x: trainable polynomial coefficients
    target_conv: precomputed fixed convolution poly_target * reversed(poly_target)
    """
    n = len(x)
    conv_x = convol_poly(x)
    constraint = conv_x + target_conv   # shape: 2n-1

    loss = (
        torch.sum(torch.abs(constraint[0:n-1])**2)
        + torch.sum(torch.abs(constraint[n:2*n])**2)
        #+ ratio * torch.abs(constraint[n-1])**2
        + (1- torch.abs(constraint[n-1]))**2
    )
    return loss


def objective_function_split(x, target_conv):
    """
    x: trainable polynomial coefficients
    target_conv: precomputed fixed convolution poly_target * reversed(poly_target)
    """
    n = len(x)
    conv_x = convol_poly(x)
    constraint = conv_x + target_conv   # shape: 2n-1

    loss = (
        torch.sum(torch.abs(constraint[0:n-1])**2)
        + torch.sum(torch.abs(constraint[n:2*n])**2)
    )
    return float(loss), float(torch.abs(constraint[n-1]))

def objective_function_entrywise(x, target_conv):
    """
    x: trainable polynomial coefficients
    target_conv: precomputed fixed convolution poly_target * reversed(poly_target)
    """
    n = len(x)
    conv_x = convol_poly(x)
    constraint = conv_x + target_conv   # shape: 2n-1

    return (torch.abs(constraint)**2).numpy()

def eval_poly_from_list(coeffs, x):
    value = 0
    for k in range(len(coeffs)):
        value += coeffs[k] * x**k
    return value

# ================================================================
# ----------------------------- MAIN -----------------------------
# ================================================================

def partner_poly(poly):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    poly_target = torch.tensor(poly, dtype=torch.float64, device=device)

    # precompute target convolution (no grad needed)
    target_conv = convol_poly(poly_target).detach()

    # initialize trainable parameters
    initial = torch.randn(len(poly_target), dtype=torch.float64, device=device, requires_grad=True)
    

    optimizer = torch.optim.LBFGS([initial], max_iter=200)
    print("Running optimization...")
    final_loss_tensor = optimizer.step(lambda: (optimizer.zero_grad(), (loss := objective_function(initial, target_conv, ratio=0.00)), loss.backward(), loss)[-1])
    final_loss = final_loss_tensor.item()
    solution = initial.detach().cpu()
    optimization_profile = objective_function_entrywise(solution, target_conv)
    renormalization = objective_function_split(solution, target_conv)[1]
    return solution.numpy(), optimization_profile, renormalization

# -----------------------------
# TEST function: plot two polynomials
# -----------------------------

def plot_two_poly(poly1, poly2):
    vals_poly_target = [eval_poly_from_list(poly1, np.exp(1j*x)) for x in np.linspace(-np.pi,np.pi,500)]
    vals_partner = [eval_poly_from_list(poly2, np.exp(1j*x)) for x in np.linspace(-np.pi,np.pi,500)]
    su2constraint = [np.abs(vals_poly_target[i])**2 + np.abs(vals_partner[i])**2 for i in range(500)]
    
    plt.plot(np.linspace(-np.pi,np.pi,500), np.abs(vals_poly_target), label="Target poly")
    plt.plot(np.linspace(-np.pi,np.pi,500), np.abs(vals_partner), label="Partner poly")
    plt.plot(np.linspace(-np.pi,np.pi,500), su2constraint, label="SU(2) constraint")
    plt.title("Polynomial evaluation on unit circle")
    plt.legend()
    plt.show()





