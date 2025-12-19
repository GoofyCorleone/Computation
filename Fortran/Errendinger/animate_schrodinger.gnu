#!/usr/bin/gnuplot
# Script de Gnuplot para animar la solución de la ecuación de Schrödinger

# Configuración general - usar terminal wxt o x11 en lugar de qt
# Intenta primero con wxt (más moderno)
if (strstrt(GPVAL_TERMINALS, 'wxt') > 0) {
    set term wxt size 1200,800 enhanced font 'Verdana,10'
} else {
    # Alternativa: usar x11 (más compatible con sistemas antiguos)
    set term x11 size 1200,800 enhanced font 'Verdana,10'
}
set encoding utf8

# Leer datos de la rejilla espacial y el potencial
set table 'pot_scaled.dat'
    plot 'grid_data.dat' using 1:($2*0.05) # Escalamos el potencial para visualizarlo mejor
unset table

# Leer número total de archivos
filename = "filecount.dat"
stats filename nooutput
num_files = int(STATS_records)

# Configuración del diseño multiplot
set multiplot layout 2,1 title "Evolución Temporal de la Ecuación de Schrödinger con Barrera de Potencial"

# Crear archivo de animación (opcional, requiere terminal gif)
# set term gif animate delay 10 size 1200,800
# set output 'schrodinger_animation.gif'

# Función para formatear el número de archivo
file_num(n) = sprintf("%05d", n)

# Bucle de animación
do for [i=0:num_files] {
    # Primer subplot: Función de onda (parte real e imaginaria)
    set origin 0,0.5
    set size 1,0.5
    set xlabel "Posición x"
    set ylabel "Amplitud"
    set title sprintf("Función de onda ψ(x,t) - Paso %d", i)
    set yrange [-0.3:0.3]
    
    datafile = sprintf("wavedata/psi_%05d.dat", i)
    
    # Extraer el tiempo del archivo de datos
    cmd = sprintf("head -1 %s | awk '{print $4}'", datafile)
    time = system(cmd)
    
    set label 1 sprintf("t = %s", time) at graph 0.02, 0.95 font ",12"
    
    # Dibujar el potencial escalado (barrera)
    set arrow 1 from 50,0.25 to 50,-0.25 nohead lw 2 lc rgb "black"
    set label 2 "Barrera\nV = 0.5" at 52,-0.2 font ",10"
    
    plot datafile using 1:2 with lines lw 2 lc rgb "blue" title "Re(ψ)", \
         datafile using 1:3 with lines lw 2 lc rgb "red" title "Im(ψ)"
    
    # Segundo subplot: Densidad de probabilidad
    set origin 0,0
    set size 1,0.5
    set xlabel "Posición x"
    set ylabel "Densidad de Probabilidad"
    set title "Densidad de Probabilidad |ψ(x,t)|²"
    set yrange [0:0.015]
    
    # Dibujar el potencial (barrera)
    set arrow 2 from 50,0 to 50,0.015 nohead lw 2 lc rgb "black"
    set label 3 "Observe el tunelamiento\na través de la barrera" at 10,0.012 font ",10"
    
    plot datafile using 1:4 with lines lw 2 lc rgb "forest-green" title "|ψ|²", \
         'pot_scaled.dat' using 1:2 with lines lc rgb "black" title "Potencial"
         
    # Pausa para la animación (ajustar para controlar velocidad)
    pause 0.1
}

# Si se está generando un GIF animado, cerrar el archivo
# set output

# Finalizar el multiplot
unset multiplot

# Mantener la ventana abierta después de la animación (sólo para terminales interactivos)
pause -1 "Presione cualquier tecla para salir..."