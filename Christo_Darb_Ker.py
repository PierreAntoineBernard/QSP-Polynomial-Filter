import numpy as np
import matplotlib.pyplot as plt
import math

def Chebyshev(degree):
    """
    Generate Chebyshev polynomial of given order and degree.

    order: int, order of the Chebyshev polynomial
    degree: int, degree of the polynomial to return

    returns: numpy array of coefficients of the polynomial of given degree
    """

    # Initialize coefficients for T_0 and T_1
    T_prev = np.zeros(degree + 1)
    T_prev[0] = 1  # T_0(x) = 1

    T_curr = np.zeros(degree + 1)
    T_curr[1] = 1  # T_1(x) = x

    poly_list = [T_prev, T_curr]

    if degree == 1:
        return poly_list

    # Recurrence relation: T_n(x) = 2*x*T_{n-1}(x) - T_{n-2}(x)
    for n in range(2, degree + 1):
        T_next = np.zeros(degree + 1)
        for d in range(degree + 1):
            if d >= 1:
                T_next[d] += 2 * T_curr[d - 1]
            if d <= degree - 2:
                T_next[d] -= T_prev[d]
        T_prev, T_curr = T_curr, T_next
        poly_list.append(T_curr)

    return poly_list

def norms_for_chebyshev(degree):
    norms = np.zeros(degree + 1)
    norms[0] = np.pi
    for n in range(1, degree + 1):
        norms[n] = np.pi / 2
    return norms

def Christoffel_Darboux_Kernel(poly_mat, norms, y):
    kernel = np.zeros_like( poly_mat[0])
    values = np.array([y**i for i in range(len(poly_mat))])

    for k in range(len(poly_mat)):
        weight = np.sum(values * poly_mat[k]) / norms[k]
        kernel += weight * poly_mat[k]
    return kernel

def eval_poly(coeffs, x):
    value = 0
    for k in range(len(coeffs)):
        value += coeffs[k] * x**k
    return value

def poly_to_laurent_transition_mat(degree):
    mat = np.zeros((degree +1, 2*degree +1))
    for i in range(2*degree +1):
        for j in range(degree +1): 
            if degree-j <= i and i <= degree + j and (i - (degree - j))%2 ==0:
                mat[j,i] = ( math.comb(j,int((i - (degree - j))/2)))/2**j  
    return mat    

def convert_poly_to_laurent(coeffs, degree):
    mat = poly_to_laurent_transition_mat(degree)
    laurent_coeffs = coeffs @ mat
    return laurent_coeffs

'''
# -----------------------------
#           EXAMPLE
# -----------------------------

poly_mat = Chebyshev(10)
norms = norms_for_chebyshev(10)
y = 0
kernel = Christoffel_Darboux_Kernel(poly_mat, norms, y)/100
laurent_coeffs = convert_poly_to_laurent(kernel, 10)



x = np.linspace(-1,1,500)
ys = np.array([eval_poly(kernel, xi) for xi in x])
plt.plot(x, ys)
plt.title('Christoffel-Darboux Kernel at y=0, real line')  
plt.show()

x = np.linspace(-np.pi,np.pi,500)
ys = np.array([np.abs(eval_poly(laurent_coeffs, np.exp(1j*xi))) for xi in x])
plt.plot(x, ys)
plt.title('Christoffel-Darboux Kernel at y=0, unit circle')  
plt.show()

print('Laurent coeffs:', laurent_coeffs)


# ================================================================
'''