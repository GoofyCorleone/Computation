import ModuloAberraciones as ma
import numpy as np

rutas1H = ['Horizontal1/{}.csv'.format(i) for i in np.arange(-20,22,2)]
rutas2H = ['Horizontal2/{}.csv'.format(i) for i in np.arange(-20,22,2)]
rutasV = ['vertical/{}.csv'.format(i) for i in np.arange(-20,22,2)]

Carpeta1 = 'AnalisisHorizontal1'
Carpeta2 = 'AnalisisHorizontal2'
Carpeta3 = 'Analisisvertical'

def AnalisisDatos(rutas, carpeta):
    aux = -20
    for ruta in rutas:
        Modulo = ma.AnalizadorShackHartmann(
            archivo_csv=ruta, 
            guardar_archivos=True, 
            nombre_carpeta=carpeta,  
            angulo=aux
        )
        Modulo.main()  
        aux += 2

if __name__ == "__main__":
    AnalisisDatos(rutasV, Carpeta3)