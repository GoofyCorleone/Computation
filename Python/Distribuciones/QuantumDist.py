import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def Q_func(n,n0,S):
    Q = (0.5*(1+n[:,0]*n0[0]+n[:,1]*n0[1]+n[:,2]*n0[2]))**(2*S)
    return Q


def plot_Q_func(n,n0,S,X,Y,Z):
    pdf_values = Q_func(n,n0,S)
    pdf_values = np.outer(pdf_values,pdf_values)
    print(np.shape(pdf_values))
    pdfnorm = Normalize(vmin=pdf_values.min(), vmax=pdf_values.max())
    normalized = pdfnorm(pdf_values)
    # print(min(normalized),max(normalized))
    print(np.shape(normalized))
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1,
                    facecolors=plt.cm.viridis(normalized),
                    # cmap = normalized,
                    linewidth=0)
    ax.set_aspect('equal')
    # ax.view_init(azim=-130, elev=0)
    # ax.axis('off')
    ax.set_title(rf"$S={S}$")

fig = plt.figure()
ax = plt.axes(projection='3d')
    
n_grid = 100
theta , phi = np.random.uniform(0,np.pi,n_grid) , np.random.uniform(0,2*np.pi,n_grid)
# theta , phi = np.linspace(0,np.pi,n_grid) , np.linspace(0,2*np.pi,n_grid)
x , y , z = np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi) , np.cos(theta)
n = np.array([x , y , z])
n = n.T
n0 = np.array([0,0,1])
n0 = n0/np.linalg.norm(n0)
S = 1
theta , phi = np.meshgrid(theta,phi)
X , Y , Z = np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi) , np.cos(theta)
plot_Q_func(n=n,n0=n0,S=S,X=X,Y=Y,Z=Z)
plt.show()