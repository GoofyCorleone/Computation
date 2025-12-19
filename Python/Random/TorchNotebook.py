import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('opencl' if torch.cuda.is_available() else 'cpu')
print(device)

a = torch.tensor([1,2,3]).to(device)
b = torch.tensor([4,5,6,7,8]).to(device)
c = torch.cat((a,b))
print(a , b , c)
print(a[0])

t = torch.linspace(-5,5,100).to(device)
x = torch.cos(t).to(device)
y = torch.sin(t).to(device)
# fig , ax = plt.figure() , plt.axes(projection = '3d')
# ax.plot3D(t.cpu(),x.cpu(),y.cpu())
# plt.show()

# Creamos un tensor de ejemplo de 3x100
tensor = torch.randn(3, 100).to(device)

# Creamos una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Hacemos un scatter plot con los valores del tensor
ax.scatter(tensor[0].cpu(), tensor[1].cpu(), tensor[2].cpu())


# Mostramos la figura
plt.show()
