import Distributions as distri
from scipy.stats import vonmises_fisher as VMF
from sphere.distribution import fb8
import sphere.distribution
import numpy as np
from scipy.optimize import fsolve

def find_kappa(rho):
    """Función de una línea para encontrar κ dado ρ"""
    return fsolve(lambda k: (1/np.tanh(k)) - (1/k) - rho, 1.0)[0]
## Para el fiteo de la distribución de VMF

route = 'DiodoPolo.csv'
try1 = distri.Distributions_Data(route=route)
# try1.PlotMD(SavePDF=False)
s1, s2 , s3 , DOP = try1.GetStokes(retS=True, retDOP=True)
samples = np.array([s1,s2,s3])
samples = samples.T
kappa_measured = find_kappa( DOP.mean() )
mu_measured = np.array([ s1.mean(), s2.mean(), s3.mean() ])
print('Vector de Stokes medio medido: ', mu_measured,'\n')
print('Valor de concentración calculado numéricamente: ', kappa_measured,'\n')

mufit , kappafit = VMF.fit(samples)
print('Vector de Stokes medio ajustado: ', mufit,'\n')
print('Valor de concentración ajustado: ', kappafit,'\n')
samples_fit = VMF(mu=mufit,kappa=kappafit).rvs(1024)
# s1f , s2f , s3f = samples_fit[:,0], samples_fit[:,1], samples_fit[:,2]
Color_DOP = VMF(mu=mufit,kappa=kappafit).pdf(samples_fit)
Color_DOP = Color_DOP / max(Color_DOP)
try1_fit = distri.Distributions_Data(DData=samples_fit)
try1_fit.PlotDD(ColorDOP=Color_DOP)

print( 'El error relativo entre el valor medio medido y el ajustado es de: ', (np.abs(mu_measured- mufit)/np.abs(mu_measured) *100).mean(),'%\n')
print( 'El error relativo entre la concentración  medida y ajustada es de: ', np.abs(kappa_measured-kappafit)/kappa_measured*100,'%\n')

samples_exp = VMF(mu=mu_measured/np.linalg.norm(mu_measured),kappa=kappa_measured).rvs(1024)
Color_DOP = VMF(mu=mu_measured/np.linalg.norm(mu_measured),kappa=kappa_measured).pdf(samples_exp)
Color_DOP = Color_DOP / max(Color_DOP)
try1_exp = distri.Distributions_Data(DData=samples_exp)
try1_exp.PlotDD(ColorDOP=Color_DOP)

## ================================================================================================================== ##
# Vamos a graficar ahora las 6 distribuciones medidas

routes = ['DiodoLedLVL{}ntensity.csv'.format(i) for i in range(1,7)]
# for route in routes:
#     dtf = distri.Distributions_Data(route=route)
#     dtf.PlotMD(SavePDF=False)

def FittingFB8(route, npts, kent=False):
    dataF = distri.Distributions_Data(route=route)
    s1 , s2 , s3 , DOP = dataF.GetStokes(retS=True, retDOP = True)
    samples_exp = np.array([s1,s2,s3])
    samples_exp = samples_exp.T
    if kent:
        k_me = sphere.distribution.kent_me(samples_exp)
    else:
        k_me = sphere.distribution.fb8_mle(samples_exp)
    print('La distribución ajustada es ', k_me,'\n')

    xs = k_me.rvs(n_samples=npts)
    pdfs = k_me.pdf(xs)
    pdfs = pdfs / max(pdfs)
    dataF_sim = distri.Distributions_Data(DData=xs)
    dataF_sim.PlotDD(ColorDOP=pdfs)


## Ajuste tipo Fisher-Binghman para los 6 niveles de potenia
for route in routes:
    FittingFB8(route=route,npts=1024,kent=False)

## Ajuste tipo Kent para los 6 niveles de potenia
for route in routes:
    FittingFB8(route=route,npts=1024,kent=True)