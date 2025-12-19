from diffractio import degrees , mm , um
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_sources_X import Scalar_source_X
from diffractio.utils_drawing import draw_several_fields
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt

# Single slit
num_data = 128
length = 250 * um
x = np.linspace(-length/2 , length/2, num_data)
wavelength = 0.6328 * um

t1 = Scalar_mask_X(x,wavelength)
t1.slit(x0=0,size=100*um)
t1.draw()
plt.show()

# Double slit

t1 = Scalar_mask_X(x , wavelength)
t1.double_slit(x0=0,size=50*um,separation=150*um)
t1.draw()
plt.show()