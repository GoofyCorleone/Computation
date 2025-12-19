import numpy as np
import  matplotlib.pyplot as plt
import  pandas as pd

rutas = ['Malus/{}.txt'.format(i) for i in np.arange(0,370,10)]

def SacarDatos(ruta):
    DataFrame = pd.read_csv(ruta,header=19, delimiter='\t')
    conteo = DataFrame['Counts per Bin '].values
    return conteo

xc = np.arange(0,370*np.pi/180,10*np.pi/180)
yc = []

x = np.linspace(0,2*np.pi,200)
y = np.cos(x)**2

for ruta in rutas:
    yc.append(SacarDatos(ruta).mean())

yc = np.array(yc)
yc = yc/max(yc)

plt.figure(1)
plt.plot(x,y, label='Curva teorica', c= 'b')
plt.scatter(xc,yc, label='Datos', c = 'r')
plt.legend()
plt.show()