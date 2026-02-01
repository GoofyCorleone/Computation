# Makefile para compilación del simulador de difracción

# Compilador
FC = gfortran
FFLAGS = -O3 -fdefault-real-8 -fdefault-double-8 -Wall -Wextra -std=f2008 -fall-intrinsics
# Para depuración:
# FFLAGS = -g -fcheck=all -fdefault-real-8 -fdefault-double-8 -Wall -Wextra -std=f2008

# Archivos fuente (en orden de dependencia)
SRCS = mod_constantes.f90 mod_integracion.f90 mod_kernels.f90 \
       mod_campos.f90 mod_simulacion.f90 mod_io.f90 main.f90

# Archivos objeto
OBJS = $(SRCS:.f90=.o)

# Ejecutable
EXEC = difraccion

# Regla por defecto
all: $(EXEC)

# Compilar el ejecutable
$(EXEC): $(OBJS)
	$(FC) $(FFLAGS) -o $@ $(OBJS)

# Regla para compilar módulos
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

# Dependencias específicas
mod_constantes.o: mod_constantes.f90
mod_integracion.o: mod_integracion.f90 mod_constantes.o
mod_kernels.o: mod_kernels.f90 mod_constantes.o
mod_campos.o: mod_campos.f90 mod_constantes.o mod_integracion.o mod_kernels.o
mod_simulacion.o: mod_simulacion.f90 mod_constantes.o mod_campos.o
mod_io.o: mod_io.f90 mod_constantes.o
main.o: main.f90 mod_constantes.o mod_simulacion.o mod_io.o

# Limpiar archivos
clean:
	rm -f $(OBJS) $(EXEC) *.mod *.bin *.txt

# Ejecutar simulación con parámetros por defecto
run: $(EXEC)
	./$(EXEC) parametros.txt resultados

# Ayuda
help:
	@echo "Opciones disponibles:"
	@echo "  make all     - Compilar el programa (por defecto)"
	@echo "  make clean   - Limpiar archivos compilados"
	@echo "  make run     - Compilar y ejecutar con parámetros por defecto"
	@echo "  make help    - Mostrar esta ayuda"

.PHONY: all clean run help