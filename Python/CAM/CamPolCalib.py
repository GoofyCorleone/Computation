from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

routeruido = 'ruido.png'
imagen = Image.open(routeruido)
Mruido = np.array(imagen)
ruido = Mruido[:,:,0]

routeI00 = 'I_0_0.bmp'
imagen = Image.open(routeI00)
MI_00 = np.array(imagen)
I_00 = MI_00[:,:,0]

routeI900 = 'I_90_0.bmp'
imagen = Image.open(routeI900)
MI_900 = np.array(imagen)
I_900 = MI_900[:,:,0]

routeI450 = 'I_45_0.bmp'
imagen = Image.open(routeI450)
MI_450 = np.array(imagen)
I_450 = MI_450[:,:,0]

routeI4590 = 'I_45_90.bmp'
imagen = Image.open(routeI4590)
MI_4590 = np.array(imagen)
I_4590 = MI_4590[:,:,0]

# routeI00 = 'CCDI(0,0).tif'
# imagen = Image.open(routeI00)
# MI_00 = np.array(imagen)
# I_00 = MI_00[:,:,0]
#
# routeI900 = 'CCDI(90,0).tif'
# imagen = Image.open(routeI900)
# MI_900 = np.array(imagen)
# I_900 = MI_900[:,:,0]
#
# routeI450 = 'CCDI(45,90).tif'
# imagen = Image.open(routeI450)
# MI_450 = np.array(imagen)
# I_450 = MI_450[:,:,0]
#
# routeI4590 = 'CCDI(90,0).tif'
# imagen = Image.open(routeI4590)
# MI_4590 = np.array(imagen)
# I_4590 = MI_4590[:,:,0]


#I_00 , I_450 , I_4590 , I_900 = I_00-ruido , I_450-ruido , I_4590-ruido, I_900-ruido

S0 = I_00 + I_900
S1 = I_00 - I_900
S2 = 2*I_450 - I_00 - I_900
S3 = 2*I_4590 - I_00 - I_900

# s0 , s1 , s2 , s3 = S1/S0 , S2/S0 , S3/S0 , S0/S0
# DOP = np.sqrt(s1**2 + s2**2 + s3**2)
# print(DOP)
# plt.imshow(s1)
# plt.show()

s1 , s2 , s3 = 0*S1 , 0*S2 , 0*S3

for i in range(0,1024):
    for j in range(0,1024):
        if S0[i,j] == 0:
            continue
        else:
            s1[i, j] = S1[i, j] / S0[i,j]
            s2[i, j] = S2[i, j] / S0[i, j]
            s3[i, j] = S3[i, j] / S0[i, j]

            # DOP = np.sqrt( s1[i,j]**2 + s2[i,j]**2 + s3[i,j]**2 )
            # if DOP == 0:
            #     continue
            # else:
            #     s1[i, j] = s1[i, j] / DOP
            #     s2[i, j] = s2[i, j] / DOP
            #     s3[i, j] = s3[i, j] / DOP

DOP = np.sqrt(s1**2 + s2**2 + s3**2)
print(s3-s1)
plt.figure(1)
plt.imshow(s2)
plt.show()
plt.figure(2)
plt.imshow(s3)
plt.show()
# if matriz_imagen.ndim == 3 and matriz_imagen.shape[2] == 3:
#     canal_rojo = matriz_imagen[:, :, 0]
#     plt.imshow(canal_rojo, cmap='gray')  # cmap='gray' para mostrar en escala de grises
#     plt.axis('off')
#     plt.title('Canal Rojo')
#     plt.show()
# else:
#     print("La imagen no es RGB o no tiene tres canales.")

