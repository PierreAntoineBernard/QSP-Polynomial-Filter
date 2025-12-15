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


# ================================================================
# ----------------------------- TEST QSP ----------------------------- 
# ================================================================

'''
P00 = Laurent_polynomials( [1], 0)
P01 = Laurent_polynomials( [0], 0)
P10 = Laurent_polynomials( [0], 0)
P11 = Laurent_polynomials( [1], 0)


for iteration in range(15):
    P00,P10 = QSP_zbasis(-0.21, P00,P10)
    P01,P11 = QSP_zbasis(-0.21, P01,P11)


P00,P10 = phase_QSP(-0.21, P00,P10)
P01,P11 = phase_QSP(-0.21, P01,P11)



#P00.check_degree()  
#P01.check_degree()  
#P10.check_degree()  
#P11.check_degree()

print('max deg P00 : ', P00.deg, ' value ', P00.coeffs[-1])
print('max deg P10 : ', P10.deg, ' value ', P10.coeffs[-1])
print('min deg P00 : ', P00.min_deg, ' value ', P00.coeffs[0])
print('min deg P10 : ', P10.min_deg, ' value ', P10.coeffs[0])


for iteration in range(15):
    angle1 = angle_from_polynomials(P00,P10)
    angle2 = angle_from_polynomials(P01,P11)

    if np.abs(angle1 - angle2) > 1e-8:
        print('Angles do not match!', angle1, angle2)
        break
    else : angle = angle1

    print('reduction :', iteration)
    P00,P10 = QSP_zbasis_inverse(angle, P00,P10)
    P01,P11 = QSP_zbasis_inverse(angle, P01,P11)

    P00.check_degree(1e-10)
    P01.check_degree(1e-10)
    P10.check_degree(1e-10)
    P11.check_degree(1e-10)

    print('# ----------------------')
    print('max deg P00 : ', P00.deg, ', min deg P00', P00.min_deg)
    print('max deg P10 : ', P10.deg, ', min deg P10', P10.min_deg)
    print('max deg P01 : ', P01.deg, ', min deg P01', P01.min_deg)
    print('max deg P11 : ', P11.deg, ', min deg P11', P11.min_deg)
    print('# ----------------------')

'''

# ================================================================
# -----------------------TEST INVERSE QSP ------------------------
# ================================================================


#Correspondence with code for partner
def Laurentpoly_from_poly(coeffs):
    new_coeffs = []
    for k in range(len(coeffs)-1):
        new_coeffs.append(coeffs[k])
        new_coeffs.append(0)
    new_coeffs.append(coeffs[-1])
    return Laurent_polynomials(new_coeffs, len(coeffs))


'''
co_poly_partner = [ 0.09420574 ,0.16790943,-0.94386086,-0.10361115,-0.0850461 ]
co_poly_target = 1j*np.array([0.08951745,0.13427618,0.04475873,0.13427618,0.08951745])




P = Laurentpoly_from_poly(co_poly_target)
Q = Laurentpoly_from_poly(co_poly_partner)

for k in range(4):
    angle1 = angle_from_polynomials(P,Q, 1)
    print('angle:',angle1)
    print('iteration:', k)

    P,Q = QSP_zbasis_inverse(angle1, P,Q)

    P.check_degree(1e-5)
    Q.check_degree(1e-5)


    print('# ----------------------')
    print('max deg P : ', P.deg, ', min deg P', P.min_deg)
    print('max deg Q : ', Q.deg, ', min deg Q', Q.min_deg)
    print('# ----------------------')


#print(square_norm_test(P,Q, 0.1))
#print(P.coeffs)
#print(Q.coeffs)


############ Plot coefficient test symmetric ##############

#print(np.array(P00.coeffs) - np.array(P00.coeffs)[::-1])

#plt.plot(range(P00.min_deg, P00.deg + 1), np.abs(P00.coeffs), 'o-')
#plt.plot(range(P10.min_deg, P10.deg + 1), np.abs(P10.coeffs), 'o-')
#plt.plot(range(P01.min_deg, P01.deg + 1), np.abs(P01.coeffs), 'o-')
#plt.plot(range(P11.min_deg, P11.deg + 1), np.abs(P11.coeffs), 'o-')
#plt.show()


############ Test unitarity ##############
#mat = matrix_from_polynomials(P00, P01, P10, P11, 1)
#print(mat)
#print((mat.conj().T) @ mat)

############ Test square norm poly ##############
#print(square_norm_test(P00, P10, 0.4))
#print(square_norm_test(P11, P01, 0.4))

'''