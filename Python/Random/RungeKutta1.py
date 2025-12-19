import numpy as np
import matplotlib.pyplot as plt

# def RK4(h,x0,n,y0,f):
#     xn , yn = x0 , np.zeros((n , len(y0)))
#     yn[0] = y0
#     Xn = np.array([x0])
#     for i in range(0,n-1):
#         k1 = h*f(xn , yn[i])
#         k2 = h*f(xn + h/2 , yn[i] + k1/2)
#         k3 = h*f(xn + h/2 , yn[i] + k2/2)
#         k4 = h*f(xn + h , yn[i] + k3)
#         yn[i + 1] = yn[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
#         xn += h
#         Xn = np.append(Xn , [xn])
#     return Xn , yn

def RK43D(h,t0,n,y0,f):
    tn = t0
    yn = np.zeros((n,len(y0)))
    Tn = np.array([t0])
    yn[0] = y0
    
    for i in range(0,n-1):
        k1 = h*f(tn , yn[i])
        k2 = h*f(tn + h/2 , yn[i] + k1/2)
        k3 = h*f(tn + h/2 , yn[i] + k2/2)
        k4 = h*f(tn + h , yn[i] + k3)
        yn[i + 1] = yn[i] + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        tn += h
        Tn = np.append(Tn,[tn])

    return Tn , yn

def E(r,t):
    return np.array([0,0,0])

def B(r,t):
    return np.array([0,1,0]) # 1 Tesla magnetic field in y direction

def RHS(t,Y):
    Ef = E(Y,t)
    Bf = B(Y,t)
    q , m =  10E-4 , 5E-3 #10e^-5 [C] , 0.0005 [Kg]
    return np.append(Y[3:],  q/m * (Ef - np.cross(Y[3:],Bf)))

y0 = np.array([0,0,0,10,1,1]) # [ r , v]
h = 0.01
t0 , tf = 0,100 #[s]
n = int((tf-t0)/h + 1)

Tr , Yr = RK43D(h,t0,n,y0,RHS)

##=====================================================================##
q , m , B = 10E-4 , 5E-3 , 1
##=====================================================================##

fig , ax = plt.figure() , plt.axes(projection = '3d')
ax.plot3D(Yr[:,0],Yr[:,1],Yr[:,2])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
plt.show()