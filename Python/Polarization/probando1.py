from main import *
import numpy as np

data, layout = Sphere3D()

# # Generate a VMF-like distribution
Jv = JonesVec(0,0)
mu = np.real(StokesFromJones(Jv))[1:]
mu = mu/np.linalg.norm(mu)
kappa = 500

## Lets use a full rotation of a birrefringent
r = np.real(StokesFromJones(JonesVec(0.3,0.5)))
delta_arr = np.linspace(0,2*np.pi,8)
mu = np.real(StokesFromJones(Jv))
for delta in delta_arr:
    mu_r = np.real(StokesFromJones(np.dot(BirrefringentQuat(r,delta,0),Jv)))
    mu_r = mu_r[1:]
    Data = vmf(mu=mu_r,kappa=kappa).rvs(200)
    x , y , z = Data[:,0] , Data[:,1] , Data[:,2]
    DistriStokes = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=3,color='blue'),showlegend=True,name='Delta{}'.format(delta))
    meanVec = go.Scatter3d(x=[0,mu_r[0]],y=[0,mu_r[1]],z=[0,mu_r[2]],mode='lines',marker=dict(size=10,color='black'),showlegend=False)
    data.append(DistriStokes)
    data.append(meanVec)    
    
rot_axis = go.Scatter3d(x=np.array([-r[1],r[1]]),y=np.array([-r[2],r[2]]),z=np.array([-r[3],r[3]]),mode='lines',marker=dict(size=15,color='red'))
data.append(rot_axis)
plotting(data=data,layout=layout)

Jv = JonesVec(0,0)
S_1 = np.real(StokesFromJones(Jv))
x , y , z = [S_1[1]] , [S_1[2]] , [S_1[3]]
stokesScatt = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=5,color='blue'),showlegend=False)
data.append(stokesScatt)
phi_arr = np.linspace(0,2*np.pi,30)
r = np.real(StokesFromJones(JonesVec(0.3,0.5)))
for phi in phi_arr:
    S =  np.real(StokesFromJones(np.dot(BirrefringentQuat(r,phi,0),Jv)))
    x , y , z = [S[1]] , [S[2]] , [S[3]]
    stokesScatt = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=5,color='blue'),showlegend=False)
    data.append(stokesScatt)
rot_axis = go.Scatter3d(x=np.array([-r[1],r[1]]),y=np.array([-r[2],r[2]]),z=np.array([-r[3],r[3]]),mode='lines',marker=dict(size=15,color='red'))
data.append(rot_axis)

plotting(data=data,layout=layout)