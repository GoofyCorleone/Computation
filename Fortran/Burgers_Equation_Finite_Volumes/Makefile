FC=gfortran

FFLAGS = -O3 -g

LNK=gfortran

OBJS = main.o evolve.o memory.o ini_arrays.o grid.o save1Ddata.o initial.o  exact.o  rhs.o  bc.o rkj.o RKDP.o FTCS.o LAXFRIEDRICHS.o LEAPFROG.o LAXWENDROFF.o 
MODS = global_parameters.o arrays.o

$(OBJS):	$(MODS)

GDS:	$(OBJS) $(MODS)
		$(LNK) $(FFLAGS) -o xOUT $(OBJS) $(MODS) 
	@ mkdir -p output
	@ mv xOUT output

.PHONY:	clean
clean:
	-rm -f *.o *.mod output/xOUT output/*.x
%.o : %.f90
	$(FC) -c $(FFLAGS) $< -o $@
