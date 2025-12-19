# Makefile para compilar el programa de simulación de la ecuación de Schrödinger
FC = gfortran
FFLAGS = -O3 -Wall -Wextra

# Nombre del programa ejecutable
EXEC = schrodinger_barrier

# Archivos fuente
SRC = schrodinger_barrier.f90

all: $(EXEC)

$(EXEC): $(SRC)
	$(FC) $(FFLAGS) -o $@ $<

clean:
	rm -f $(EXEC) *.o *.mod

# Eliminar también todos los archivos de datos generados
cleanall: clean
	rm -rf wavedata/ *.dat

run: $(EXEC)
	./$(EXEC)

# Meta para ejecutar la simulación y luego la animación
animate: run
	gnuplot animate_schrodinger.gnu

.PHONY: all clean cleanall run animate