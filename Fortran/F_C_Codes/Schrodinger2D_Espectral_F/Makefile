FC=gfortran

FFLAGS = -O3 -g

LNK=gfortran

OBJS = main.o memory.o Vecs.o saving.o
MODS = lu.o arrays.o Global_parameters.o funcmod.o

$(OBJS):	$(MODS)

SchroEspec:	$(OBJS) $(MODS)
		$(LNK) $(FFLAGS) -o xOUT $(OBJS) $(MODS) 
	@ mkdir -p output
	@ mv xOUT output

.PHONY: clean
clean:
	-rm -f *.o *.mod output/xOUT output/*.x
%.o : %.f90
	$(FC) -c $(FFLAGS) $< -o $@