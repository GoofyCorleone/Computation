import numpy as np
from plotly import graph_objects as go
from plotly.offline import iplot
from scipy.stats import vonmises_fisher as vmf
from sphere.distribution import fb8

def Sphere3D():
    phi , theta = np.linspace(0,2*np.pi,200) , np.linspace(0,np.pi,200)
    phi , theta = np.meshgrid(phi,theta)
    x , y , z = np.sin(theta)*np.cos(phi) , np.sin(theta)*np.sin(phi) , np.cos(theta)
    
    sphere = go.Surface(
        x=x,
        y=y,
        z=z,
        showscale=False,
        colorscale='Peach',
        opacity=0.2
    )
    
    layout = go.Layout(
        title='Poincaré Sphehre',
        scene=dict(
            xaxis_title='S1',
            yaxis_title='S2',
            zaxis_title='S3',
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False
        )
    )
    
    lineS1 = go.Scatter3d(x=[-1,1],y=[0,0],z=[0,0],mode='lines',line=dict(color='black',width=3),showlegend=False)
    lineS2 = go.Scatter3d(x=[0,0],y=[-1,1],z=[0,0],mode='lines',line=dict(color='black',width=3),showlegend=False)
    lineS3 = go.Scatter3d(x=[0,0],y=[0,0],z=[-1,1],mode='lines',line=dict(color='black',width=3),showlegend=False)
    
    data = [sphere,lineS1,lineS2,lineS3]
    return data, layout

def plotting(data,layout):
    fig = go.Figure(data = data,layout=layout)
    fig.update_layout(
        scene=dict(
            annotations=[
                dict(
                    showarrow=False,
                    x=0,
                    y=0,
                    z=1,
                    text = 'S3'
                ),
                dict(
                    showarrow=False,
                    x=0,
                    y=1,
                    z=0,
                    text='S2'
                ),
                dict(
                    showarrow=False,
                    x=1,
                    y=0,
                    z=0,
                    text='S1'
                )
            ]
        )
    )
    iplot(fig)

def StokesVec(alpha,chi):
    S1 = np.cos(2*alpha)*np.cos(2*chi)
    S2 = np.cos(2*alpha)*np.sin(2*chi)
    S3 = np.sin(2*chi)
    return np.array([S1,S2,S3])

def BasisChange(alpha=1,chi=1):
    T = np.array(
        [
            [np.cos(alpha)*np.cos(chi) -np.sin(alpha)*np.sin(chi)*1j , np.cos(alpha)*np.sin(chi) + 1j*np.sin(alpha)*np.cos(chi)],
            [np.sin(alpha)*np.cos(chi) +np.cos(alpha)*np.sin(chi)*1j , np.sin(alpha)*np.sin(chi) - 1j*np.cos(alpha)*np.cos(chi)]
        ]
    )
    S = np.array([np.cos(alpha)*np.cos(chi) -np.sin(alpha)*np.sin(chi)*1j, np.sin(alpha)*np.cos(chi) +np.cos(alpha)*np.sin(chi)*1j])
    return T , np.dot(np.linalg.inv(T),S)

def Birrefringent(T,delta,basis='eliptic'):
    R_delta = np.array(
        [
            [np.exp(1j*delta) , 0],
            [0 , 1]
        ]
    )
    
    if basis == 'eliptic':
        return R_delta
    elif basis != 'eliptic':
        return np.dot(np.linalg.inv(T),np.dot(R_delta,T))

def BirrefringentQuat(S,delta,theta):
    sigma0,sigma1,sigma2,sigma3 = np.array([[1,0],[0,1]]),np.array([[1,0],[0,-1]]),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]])
    Sigma = np.array([sigma0,sigma1,sigma2,sigma3])
    Rot = np.array(
        [
            [np.cos(theta) , np.sin(theta)],
            [-np.sin(theta) , np.cos(theta)]
        ]
    )
    # R = np.exp(1j*np.dot(Sigma.T,S).T*delta/2)
    R = np.cos(delta/2)*+sigma0 + 1j*np.sin(delta/2)*np.dot(Sigma[1:].T,S[1:])
    R =  np.dot(np.linalg.inv(Rot),np.dot(R,Rot))
    return R

def JonesVec(alpha,chi):
    JV = np.array([np.cos(alpha)*np.cos(chi) - 1j*np.sin(alpha)*np.sin(chi),
                     np.sin(alpha)*np.cos(chi)+1j*np.cos(alpha)*np.sin(chi)])
    return JV

def StokesFromJones(JV):
    J = np.outer(JV,np.conjugate(JV))
    S0,S1,S2,S3 = J[0,0]+J[1,1], J[0,0]-J[1,1], J[0,1]+J[1,0] , 1j*(J[1,0]-J[0,1])
    S = np.array([S0,S1,S2,S3])
    return S/S0

def GetMueller(T):
    sigma0,sigma1,sigma2,sigma3 = np.array([[1,0],[0,1]]),np.array([[1,0],[0,-1]]),np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]])
    Sigma = np.array([sigma0,sigma1,sigma2,sigma3])
    M = np.zeros((4,4))
    for i in range(0,4):
        for k in range(0,4):
            a = np.dot(np.conjugate(T).T,Sigma[i])
            b = np.dot(T,Sigma[k])
            M[i,k] = 0.5*np.trace(np.dot(a,b))
    return M


data, layout = Sphere3D()

# phiR, thetaR = np.random.uniform(0,2*np.pi,10) , np.random.uniform(0,np.pi,10)
# x , y , z = np.sin(thetaR)*np.cos(phiR) , np.sin(thetaR)*np.sin(phiR) , np.cos(thetaR)
# JV = JonesVec(0,0)
# S = StokesFromJones(JV)
# S = np.real(S)
# x , y , z = np.array([S[1]]),np.array([S[2]]),np.array([S[3]])

# r = np.real(StokesFromJones(JonesVec(np.pi/32,np.pi/9)))
# print(r)

# Rotated state changing birrefringence
# S = StokesFromJones(np.dot(BirrefringentQuat(r,np.pi/2,0),JV))
# S = np.real(S)
# print(S)
# x , y , z = np.append(x,S[1]) , np.append(y,S[2]) , np.append(z,S[3])


# Complete round
# phi = np.linspace(0,2*np.pi,30)
# x , y , z = [] , [] , []
# for item in phi:
#     S = np.real(StokesFromJones(np.dot(BirrefringentQuat(r,item,0),JV)))
#     x.append(S[1])
#     y.append(S[2])
#     z.append(S[3])
# x , y , z = np.array(x) , np.array(y) , np.array(z)

# theta = np.linspace(0,2*np.pi,30)
# x , y , z = [] , [] , []
# for item in theta:
#     S = np.real(StokesFromJones(np.dot(BirrefringentQuat(r,np.pi/2,item),JV)))
#     x.append(S[1]);y.append(S[2]);z.append(S[3])

# rot_axis = go.Scatter3d(x=np.array([-r[1],r[1]]),y=np.array([-r[2],r[2]]),z=np.array([-r[3],r[3]]),mode='lines',marker=dict(size=15,color='red'))
# stokesScatt = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=5,color='blue'),showlegend=False)
# data.append(stokesScatt)
# data.append(rot_axis)


# plotting(data=data,layout=layout)

# # Generate a VMF-like distribution
# Jv = JonesVec(0.3,0.2)
# mu = np.real(StokesFromJones(Jv))[1:]
# mu = mu/np.linalg.norm(mu)
# kappa = 500
# Data = vmf(mu=mu,kappa=kappa).rvs(200)
# x , y , z = Data[:,0] , Data[:,1] , Data[:,2]
# DistriStokes = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=3,color='red'),showlegend=True,name='original')
# meanVec = go.Scatter3d(x=[0,mu[0]],y=[0,mu[1]],z=[0,mu[2]],mode='lines',marker=dict(size=10,color='black'),showlegend=False)
# data.append(DistriStokes)
# data.append(meanVec)

# # # operate all the accesible states
# r = np.real(StokesFromJones(JonesVec(0.1,0.9)))
# M = GetMueller(BirrefringentQuat(r,np.pi/2,0))

# DData = np.ones((4,len(Data)))
# DData[1:,:] = Data.T
# # Data = np.dot(M,DData).T[:,1:]
# Data = np.dot(M,DData)[1:,:].T
# x , y , z = Data[:,0] , Data[:,1] , Data[:,2]
# # x , y , z = Data[0,:] , Data[1,:] , Data[2,:]
# DistriStokes = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=3,color='blue'),showlegend=True,name='Mueller')
# mu_fit , kappa_fit = vmf.fit(Data)
# meanVec = go.Scatter3d(x=[0,mu_fit[0]],y=[0,mu_fit[1]],z=[0,mu_fit[2]],mode='lines',marker=dict(size=10,color='black'),showlegend=False)
# rotAxis = go.Scatter3d(x=[-r[1],r[1]],y=[-r[2],r[2]],z=[r[3],-r[3]],mode='lines',marker=dict(size=15,color='black'),showlegend=False)
# data.append(DistriStokes)
# data.append(meanVec)
# data.append(rotAxis)

# # # Operate only the mean vector of the original distribution
# Jv = np.dot(BirrefringentQuat(r,np.pi/2,0),Jv)
# mu = np.real(StokesFromJones(Jv))[1:]
# mu = mu/np.linalg.norm(mu)
# kappa = 500
# Data = vmf(mu=mu,kappa=kappa).rvs(200)
# x , y , z = Data[:,0] , Data[:,1] , Data[:,2]
# DistriStokes = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=3,color='green'),showlegend=True,name='Jones')
# meanVec = go.Scatter3d(x=[0,mu[0]],y=[0,mu[1]],z=[0,mu[2]],mode='lines',marker=dict(size=10,color='black'),showlegend=False)
# data.append(DistriStokes)
# data.append(meanVec)

# # # Transformation law from Stokes-Mueller law
# r = np.real(StokesFromJones(JonesVec(0.1,0.9)))
# M = GetMueller(BirrefringentQuat(r,np.pi/2,0))
# mu = np.real(StokesFromJones(Jv))
# mu_prima = np.dot(M,mu)
# mu_prima = mu_prima[1:]
# kappa = 500
# mu_prima = mu_prima/np.linalg.norm(mu_prima)
# Data = vmf(mu=mu_prima,kappa=kappa).rvs(200)
# x , y , z = Data[:,0] , Data[:,1] , Data[:,2]
# DistriStokes = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=3,color='magenta'),showlegend=True,name='Stokes_Mueller')
# meanVec = go.Scatter3d(x=[0,mu_prima[0]],y=[0,mu_prima[1]],z=[0,mu_prima[2]],mode='lines',marker=dict(size=10,color='black'),showlegend=False)
# data.append(DistriStokes)
# data.append(meanVec)


# plotting(data=data,layout=layout)

# distri = fb8(np.pi/16,-2.5*np.pi/1,0,0.00001,1,0.0001,0.5,0.3)
# phiR, thetaR = np.random.uniform(0,2*np.pi,700) , np.random.uniform(0,np.pi,700)
# xs = distri.spherical_coordinates_to_nu(*np.array([thetaR,phiR]))
# pdfs= distri.pdf(xs)
# indices = []
# for i in range(0,len(pdfs)):
#     if pdfs[i] < (pdfs.mean()):
#         indices.append(i)
# print(max(pdfs))
# xs = np.delete(xs,indices,0)
# z , x , y = xs.T
# DistriStokes = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=3,color=pdfs,colorscale='viridis',colorbar=dict(title='PD'),cmin=0,cmax=1),showlegend=True,name='FB8')

distri = fb8(np.pi/16,-2.5*np.pi/1,0,47,59,-1).rvs(200)
z , x , y = distri[:,0] , distri[:,1] , distri[:,2]
DistriStokes = go.Scatter3d(x=x,y=y,z=z,mode='markers',marker=dict(size=3,color='blue'),showlegend=True,name='FB8')
data.append(DistriStokes)
plotting(data=data,layout=layout)