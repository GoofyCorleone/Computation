import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# Definir la función C(x)
def C(x):
    ln_x = np.log(x)
    numerator = 20 * (ln_x)
    denominator = x * (ln_x**2+1)*(ln_x+3)
    return numerator / denominator
def C2(x):
    numerador = 20
    denominator = 1 + 3*np.exp(-3*x)
    return  numerador / denominator

# Límites de integración
a = 2  # En kilómetros
b = 502  # En kilómetros

# Calcular el costo acumulado usando integración numérica
CA, error = quad(C, a, b)
print('CA=' , CA)
print('\nError acumulado: ', error)