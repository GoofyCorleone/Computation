import numpy as np
import matplotlib.pyplot as plt
from scipy.special import iv

def VonMisesFisherPlot(p,kappa,mu,x):
    c = kappa**(p/2-1)/((2*np.pi)**(p/2) * iv(p/2-1,kappa))
    # c = kappa/(4*np.pi*np.sinh(kappa))
    return c * np.exp(kappa * np.dot(x,mu))

# theta , phi = np.linspace(0,np.pi,100) , np.linspace(0,2*np.pi,100)
theta , phi = np.random.uniform(0, np.pi, 500) , np.random.uniform(0, 2 * np.pi, 500)
x , y , z = np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi) , np.cos(theta)
x_arr = np.array([x,y,z]).T
mu = np.array([0,0,1])
mu = mu/np.linalg.norm(mu)
p , kappa = 3 ,0.0001
VMF = VonMisesFisherPlot(p,kappa,mu,x_arr)

theta , phi = np.linspace(0,np.pi,200) , np.linspace(0,2*np.pi,200)
THETA , PHI = np.meshgrid(theta,phi)
X , Y , Z = np.sin(THETA)*np.cos(PHI) , np.sin(THETA)*np.sin(PHI) , np.cos(THETA)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,Z,color='b',alpha=0.1)
sc = ax.scatter(x,y,VMF, c=VMF, cmap='viridis',s=5)

fig.colorbar(sc, label='PDF')
plt.axis('off')
ax.set_box_aspect([1,1,1])
plt.show()

# print(x)
# print(mu)
# print(x_arr.T)