import numpy as np 
import matplotlib.pyplot as plt
from scipy import constants

def B_dipolar(r):
    m = np.array([1,2,3])
    R = np.linalg.norm(r)
    B = (constants.mu_0)/(4*np.pi)*(3*np.dot(m,r)*r - R**2 * m)/R**5
    return B