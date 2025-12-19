import numpy as np
import plotly.graph_objects as go

# def devs(t,Ex,Ey,N,n):
#     I = (Ex*np.conjugate(Ex))**2  + (Ey*np.conjugate(Ey))**2
#     DtEx = kappa*(1 + 1j*alpha)*((N-1)*Ex + 1j*n*Ey) - (gamma_alpha +1j*gamma_p)*Ex
#     DtEy = kappa * (1 + 1j * alpha) * ((N - 1) * Ey + 1j * n * Ex) - (gamma_alpha + 1j * gamma_p) * Ey
#     DtN = -gamma*(N*(1 + I) - mu + 1j*n*( Ey*np.conjugate(Ex) - Ex*np.conjugate(Ey) ))
#     Dtn = -gamma_s*n - gamma*(n*I + 1j*N*(Ey*np.conjugate(Ex) - Ex*np.conjugate(Ey)))
#
#     return DtEx , DtEy , DtN , Dtn

def f1(t,Ex,Ey,N,n):
    DtEx = kappa*(1 + 1j*alpha)*((N-1)*Ex + 1j*n*Ey) - (gamma_alpha +1j*gamma_p)*Ex
    return  DtEx
def f2(t,Ex,Ey,N,n):
    DtEy = kappa * (1 + 1j * alpha) * ((N - 1) * Ey + 1j * n * Ex) - (gamma_alpha + 1j * gamma_p) * Ey
    return DtEy
def f3(t,Ex,Ey,N,n):
    I = (Ex * np.conjugate(Ex)) ** 2 + (Ey * np.conjugate(Ey)) ** 2
    DtN = -gamma * (N * (1 + I) - mu + 1j * n * (Ey * np.conjugate(Ex) - Ex * np.conjugate(Ey)))
    return DtN
def f4(t,Ex,Ey,N,n):
    I = (Ex * np.conjugate(Ex)) ** 2 + (Ey * np.conjugate(Ey)) ** 2
    Dtn = -gamma_s * n - gamma * (n * I + 1j * N * (Ey * np.conjugate(Ex) - Ex * np.conjugate(Ey)))
    return Dtn

def rk4_step(Ex,Ey,N,n,t, h):
    F1 , F2 , F3 , F4 = f1(t,Ex,Ey,N,n) , f2(t,Ex,Ey,N,n) , f3(t,Ex,Ey,N,n) , f4(t,Ex,Ey,N,n)
    k1_Ex , k1_Ey , k1_N , k1_n = h*F1 , h*F2 , h*F3 , h*F4

    F1 = f1(t + 0.5*h,Ex+0.5*k1_Ex,Ey+0.5*k1_Ey,N+0.5*k1_N,n+0.5*k1_n)
    F2 = f2(t + 0.5*h,Ex+0.5*k1_Ex,Ey+0.5*k1_Ey,N+0.5*k1_N,n+0.5*k1_n)
    F3 = f3(t + 0.5*h,Ex+0.5*k1_Ex,Ey+0.5*k1_Ey,N+0.5*k1_N,n+0.5*k1_n)
    F4 = f4(t + 0.5*h,Ex+0.5*k1_Ex,Ey+0.5*k1_Ey,N+0.5*k1_N,n+0.5*k1_n)
    k2_Ex, k2_Ey, k2_N, k2_n = h * F1, h * F2, h * F3, h * F4

    F1 = f1(t + 0.5 * h, Ex + 0.5 * k2_Ex, Ey + 0.5 * k2_Ey, N + 0.5 * k2_N, n + 0.5 * k2_n)
    F2 = f2(t + 0.5 * h, Ex + 0.5 * k2_Ex, Ey + 0.5 * k2_Ey, N + 0.5 * k2_N, n + 0.5 * k2_n)
    F3 = f3(t + 0.5 * h, Ex + 0.5 * k2_Ex, Ey + 0.5 * k2_Ey, N + 0.5 * k2_N, n + 0.5 * k2_n)
    F4 = f4(t + 0.5 * h, Ex + 0.5 * k2_Ex, Ey + 0.5 * k2_Ey, N + 0.5 * k2_N, n + 0.5 * k2_n)
    k3_Ex, k3_Ey, k3_N, k3_n = h * F1, h * F2, h * F3, h * F4

    F1 = f1(t + h, Ex + k3_Ex, Ey + k3_Ey, N + k3_N, n + k3_n)
    F2 = f2(t + h, Ex + k3_Ex, Ey + k3_Ey, N + k3_N, n + k3_n)
    F3 = f3(t + h, Ex + k3_Ex, Ey + k3_Ey, N + k3_N, n + k3_n)
    F4 = f4(t + h, Ex + k3_Ex, Ey + k3_Ey, N + k3_N, n + k3_n)
    k4_Ex, k4_Ey, k4_N, k4_n = h * F1, h * F2, h * F3, h * F4

    Ex += (k1_Ex + 2*k2_Ex + 2*k3_Ex + k4_Ex) / 6
    Ey += (k1_Ey + 2 * k2_Ey + 2 * k3_Ey + k4_Ey) / 6
    N += (k1_N + 2 * k2_N + 2 * k3_N + k4_N) / 6
    n += (k1_n + 2 * k2_n + 2 * k3_n + k4_n) / 6

    return  Ex , Ey , N , n

mu = 2.60         # Normalized thresold current
alpha = 3         # Linedwidth enhancement factor
kappa = 300       # Optical field decay rate
gamma_p = 30      # Phase anisotropy by linear birefringence
gamma_alpha = 0.5 # Amplitude anisotropy by linear dichroism
gamma = 1         # Decay rate of N
gamma_s = 0.50    # Spin-flip relaxation rate

Ex0 = 1 + 0.0j      # Initial value of Ex
Ey0 = 0.1 + 3.5j      # Initial value of Ey
N0 = 10           # Initial value of N
n0 = 0.7           # Initial value of n
t = 0            # Initial time

h = 0.0005        # Temporal resolution
t_final = 10      # Final time in [ns]
num_steps = int(t_final / h)
Integration = np.zeros((num_steps,5),dtype=complex)

for step in range(1,num_steps):
    Ex0 , Ey0 , N0 , n0 = rk4_step(Ex0,Ey0,N0,n0,t,h)
    t += h
    Integration[step-1,:] = np.array([Ex0 , Ey0 , N0 , n0 , t])

Ex_vals = Integration[:, 0]
Ey_vals = Integration[:, 1]
N_vals = Integration[:, 2]
n_vals = Integration[:, 3]
times = Integration[:, 4]

S0 = np.real( Ex_vals * np.conjugate(Ex_vals) + Ey_vals * np.conjugate(Ey_vals) )
S1 = np.real( Ex_vals * np.conjugate(Ex_vals) - Ey_vals * np.conjugate(Ey_vals) )
S2 = np.real( Ex_vals * np.conjugate(Ey_vals) + Ey_vals * np.conjugate(Ex_vals) )
S3 = np.real( 1j* (Ex_vals * np.conjugate(Ey_vals) - Ey_vals * np.conjugate(Ex_vals)) )

S0 , S1 , S2 , S3 = S0[S0 > 0] , S1[S0 > 0] , S2[S0 > 0] , S3[S0 > 0]
s1 , s2 , s3 = S1/S0 , S2/S0 , S3/S0


num_puntos = len(s1)
x,y,z = s1 , s2 , s3

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

fig = go.Figure()

fig.add_trace(go.Surface(x=x_sphere, y=y_sphere, z=z_sphere, opacity=0.5, colorscale='Blues', showscale=False))

fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=5, color='red'), name='Distribución de estados de polarizacioń'))

fig.add_trace(go.Scatter3d(x=x, y=y, z=np.ones(num_puntos)*2, mode='markers', marker=dict(size=5, color='green'), name='Proyección en el plano S1-S2'))

fig.add_trace(go.Scatter3d(x=x, y=np.ones(num_puntos)*2, z=z, mode='markers', marker=dict(size=5, color='blue'), name='Proyección en el plano S1-S3'))

fig.add_trace(go.Scatter3d(x=np.ones(num_puntos)*2, y=y, z=z, mode='markers', marker=dict(size=5, color='purple'), name='Proyección en el plano S2-S3'))

fig.update_layout(scene=dict(
    xaxis=dict(title='S1'),
    yaxis=dict(title='S2'),
    zaxis=dict(title='S3')
))

fig.show()