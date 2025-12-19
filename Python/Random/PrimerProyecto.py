import numpy as np
import matplotlib.pyplot as plt

##========================== Functions definitions ==================================#

def E(r=[],t=0):
    """
        Electric field function: Here we're goingo to define the electric field vector
        Which can depend of the position and the time.
    """
    return np.array([0,0,0])

def B(r=[],t=0):
    """
        Magnetic field function: Here we're goingo to define the electric field vector
        Which can depend of the position and the time.
    """
    return np.array([0,1,0])

def RHS(t,Y):
    """
    Right hand side of the Runge-Kutta and the Lorentz force
    Where:
        -Y 
    This has got the form:
    RHS = [x',y',z',x,y,z]
    """
    m , q = 1 , 100*(-1.6E-19)
    Ev , Bv = E() , B()
    return np.append(Y[3:] , q/m*(Ev + np.cross(Y[3:],Bv)))

def RK4(RHS,h,tn,yn,n):
    Y = np.zeros((n , 6))
    Y[0] = yn
    for i in range(0, n - 1):
        k1 = h * RHS(tn , Y[i])
        k2 = h * RHS(tn + 0.5*h , Y[i] + 0.5*k1)
        k3 = h * RHS(tn + 0.5*h , Y[i] + 0.5*k2)
        k4 = h * RHS(tn + h , Y[i] + k3)
        tn += h
        Y[i +1] = Y[i] + k1/6 + k2/3 + k3/3 + k4/6
        
    return Y

##========================== Initial data and parameters ==================================#

R = np.array([0,0,0,1,0,0])
t0 , tf , h = 0 , 50 , 0.001
nt = int((tf-t0)/h + 1)

sol = RK4(RHS,h,t0,R,nt)
print(sol)

fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot3D(sol[:,0] , sol[:,1] , sol[:,2] , label = 'Solución numérica')
ax.set_xlabel('x [m]')
ax.set_ylabel('Y [m]')
plt.show()