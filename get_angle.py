import matplotlib.pyplot as plt
import numpy as np

#Create a class for Laurent polynomials with
# - list of the coefficients
# - min and max degree
# - methods for evaluation, multiplication/division by x, multiplication by scalar
class Laurent_polynomials:
    def __init__(self, coefficients, deg):
        self.coeffs = coefficients
        self.deg = deg
        self.min_deg = deg - len(coefficients) + 1

    def evaluate(self,x):
        value = 0
        for k in range(len(self.coeffs)):
            value += self.coeffs[k] * x**(self.min_deg + k)
        return value

    def divide_by_x(self):
        self.deg -= 1
        self.min_deg -= 1

    def multiply_by_x(self):
        self.deg += 1
        self.min_deg += 1

    def multiply_coeff(self, a):
        self.coeffs = self.coeffs * a

    def check_degree(self, eps=0.001):
        while np.abs(self.coeffs[0]) <= eps:
            del self.coeffs[0]
            self.min_deg += 1

        while np.abs(self.coeffs[-1]) <= eps:
            del self.coeffs[-1]
            self.deg -= 1

# -----------------------------
# ADDITION OF LAURENT POLYNOMIALS
# -----------------------------
def add_polynomials(P1, P2, a, b):
    min_deg = min(P1.min_deg, P2.min_deg)
    max_deg = max(P1.deg, P2.deg)
    new_coeffs = []
    for k in range(min_deg, max_deg + 1):
        coeff1 = a*P1.coeffs[k - P1.min_deg] if P1.min_deg <= k <= P1.deg else 0
        coeff2 = b*P2.coeffs[k - P2.min_deg] if P2.min_deg <= k <= P2.deg else 0
        new_coeffs.append(coeff1 + coeff2)
    return Laurent_polynomials(new_coeffs, max_deg)

# -----------------------------
# QSP for Layer transformation on Laurent polynomials
# (walk operator in z-basis and phase operator in x-basis)
# -----------------------------
def QSP_zbasis(theta, poly1,poly2):
    new_poly1 = add_polynomials(poly1, poly2, np.cos(theta), 1j*np.sin(theta))
    new_poly2 = add_polynomials(poly1,poly2, 1j*np.sin(theta), np.cos(theta))

    new_poly1.divide_by_x()
    new_poly2.multiply_by_x()
    return new_poly1, new_poly2

# Last layer (additional phase transformation)
def phase_QSP(theta, poly1,poly2):
    new_poly1 = add_polynomials(poly1, poly2, np.cos(theta), 1j*np.sin(theta))
    new_poly2 = add_polynomials(poly1,poly2, 1j*np.sin(theta), np.cos(theta))
    return new_poly1, new_poly2

# Inverse QSP for Layer transformation on Laurent polynomials
def inverse_QSP_zbasis(theta, poly1,poly2):
    poly1.multiply_by_x()
    poly2.divide_by_x()
    
    new_poly1 = add_polynomials(poly1, poly2, np.cos(theta), -1j*np.sin(theta))
    new_poly2 = add_polynomials(poly1,poly2, -1j*np.sin(theta), np.cos(theta))

    return new_poly1, new_poly2

# ----------------------------
# QSP Inversion
# ----------------------------

def angle_from_polynomials(P,Q, type = 0, alert = False):
    P.check_degree( 1e-10)
    Q.check_degree(1e-10)

    P_leading = P.coeffs[-1]
    Q_leading = Q.coeffs[-1]
    P_last = P.coeffs[0]
    Q_last = Q.coeffs[0]

    theta = np.arctan(- 1j * P_leading / Q_leading)
    thetap = np.arctan(- 1j * Q_last / P_last)

    if alert:
        if np.abs(theta - thetap) > 1e-4:
            print('Warning: angles do not match!:',np.abs(theta - thetap) )
    if type == 0: return theta
    else : return thetap

def QSP_zbasis_inverse(theta, poly1,poly2):
    new_poly1 = add_polynomials(poly1, poly2, np.cos(theta), -1j*np.sin(theta))
    new_poly2 = add_polynomials(poly1,poly2, -1j*np.sin(theta), np.cos(theta))

    new_poly1.multiply_by_x()
    new_poly2.divide_by_x()
    return new_poly1, new_poly2


# -----------------------------
# CHANGE OF BASIS FUNCTIONS
# -----------------------------

# Goes from representation walk as Z rotation to Y rotation
# Phase for control from X rotation to Z rotation
def change_basis_polynomials(P00, P01, P10, P11):
    Q00 = add_polynomials(P00, P01, (1-1j)/2, (1-1j)/2)
    Q01 = add_polynomials(P00, P01, (-1-1j)/2, (1+1j)/2)
    Q10 = add_polynomials(P10, P11, (1-1j)/2, (1-1j)/2)
    Q11 = add_polynomials(P10, P11,(-1-1j)/2, (1+1j)/2)

    R00 = add_polynomials(Q00, Q10, (1+1j)/2,(1+1j)/2)
    R01 = add_polynomials(Q01, Q11, (1+1j)/2,(1+1j)/2)
    R10 = add_polynomials(Q00, Q10, (-1+1j)/2,(1-1j)/2)
    R11 = add_polynomials(Q01, Q11, (-1+1j)/2,(1-1j)/2)

    return R00, R01, R10, R11

# Goes from representation walk as Y rotation to Z rotation
# Phase for control from Z rotation to X rotation
def inverse_change_basis_polynomials(P00, P01, P10, P11):
    Q00 = add_polynomials(P00, P01, (1+1j)/2, (-1+1j)/2)
    Q10 = add_polynomials(P10, P11, (1+1j)/2, (-1+1j)/2)
    Q01 = add_polynomials(P00, P01, (1+1j)/2, (1-1j)/2)
    Q11 = add_polynomials(P10, P11, (1+1j)/2, (1-1j)/2)

    R00 = add_polynomials(Q00, Q10, (1-1j)/2,(-1-1j)/2)
    R01 = add_polynomials(Q00, Q10, (1-1j)/2,(1+1j)/2)
    R10 = add_polynomials(Q01, Q11, (1-1j)/2,(-1-1j)/2)
    R11 = add_polynomials(Q01, Q11, (1-1j)/2,(1+1j)/2)

    return R00, R01, R10, R11

# -----------------------------
# TEST FUNCTIONS    
# -----------------------------

# Function to obtain final QSP matrix for a given x
def matrix_from_polynomials(P00, P01, P10, P11, x):
    a = P00.evaluate(x)
    b = P01.evaluate(x)
    c = P10.evaluate(x)
    d = P11.evaluate(x)
    return np.array([[a, b],[c, d]])

# Function to test the su(2) unitarity condition
def square_norm_test(poly1,poly2, theta):
    return np.abs(poly1.evaluate(np.exp(1j*theta)))**2 + np.abs(poly2.evaluate(np.exp(1j*theta)))**2


#Correspondence with code for partner
def Laurentpoly_from_poly(coeffs):
    new_coeffs = []
    for k in range(len(coeffs)-1):
        new_coeffs.append(coeffs[k])
        new_coeffs.append(0)
    new_coeffs.append(coeffs[-1])
    return Laurent_polynomials(new_coeffs, len(coeffs))


