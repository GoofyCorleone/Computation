import diffractsim
# diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField,ApertureFromImage, cf, mm, cm, nm, m

F = MonochromaticField(
    wavelength = 632.8 * nm, extent_x=20. * mm, extent_y=20. * mm, Nx=2048, Ny=2048, intensity =2
)

F.add(ApertureFromImage("/Users/carlo/Downloads/Brayan/Brayan/CuadriculaEnfocada.jpg", image_size=(2 * cm, 2 * cm), simulation = F))


# F.propagate(z=1000*m)
F.zoom_propagate(80*cm, x_interval = [-10 * cm, 10 * cm], y_interval = [-10*cm, 10*cm])  # Dim senso


# rgb =F.get_colors()
# F.plot_colors(rgb, xlim=[-7* mm, 7* mm], ylim=[-7* mm, 7* mm])
I = F.get_intensity()
F.plot_intensity(I, square_root = True, units = mm, grid = True, figsize = (14,5), slice_y_pos = 0*mm, dark_background = False)



# import diffractsim
# # diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

# from diffractsim import MonochromaticField,ApertureFromImage, cf, mm, cm

# F = MonochromaticField(
#     spectrum=2 * cf.illuminant_d65, extent_x=18 * mm, extent_y=18 * mm, Nx=1024, Ny=1024
# )

# F.add(ApertureFromImage("/Users/carlo/Downloads/Brayan/Brayan/CuadriculaEnfocada.jpg", image_size=(2 * cm, 2 * cm), simulation = F))

# F.propagate(z=80*cm)

# I = F.get_intensity()
# F.plot_intensity(I, square_root = True, units = mm, grid = True, figsize = (14,5), slice_y_pos = 0*mm, dark_background = False)

# # rgb =F.get_colors()
# # F.plot_colors(rgb, xlim=[-7* mm, 7* mm], ylim=[-7* mm, 7* mm])