# Gnuplot script (plot.plt)
set title "Parabola Plot from Fortran Data"
set xlabel "X value"
set ylabel "Y value"
set grid
plot "data.txt" using 1:2 with lines title "y = x^2"
pause -1 "Press any key to exit"
