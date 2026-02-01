!===============================================================================
! PROGRAMA: main
! Propósito: Programa principal para simulación de difracción
!===============================================================================
program difraccion_main
    use mod_constantes
    use mod_simulacion
    use mod_io
    implicit none
    
    ! Variables
    type(parametros_sistema) :: params
    type(resultados_simulacion) :: results
    integer :: Nx, Ny
    real(dp) :: x_range, y_range
    character(len=200) :: archivo_parametros, archivo_resultados
    
    ! Encabezado
    print *, "=================================================="
    print *, "SIMULADOR DE DIFRACCION 2D - FORTRAN 90"
    print *, "=================================================="
    print *, ""
    
    ! Obtener nombres de archivos desde línea de comandos
    if (command_argument_count() < 2) then
        archivo_parametros = "parametros.txt"
        archivo_resultados = "resultados_simulacion"
        print *, "Uso: ./difraccion [parametros.txt] [resultados]"
        print *, "Usando archivos por defecto"
    else
        call get_command_argument(1, archivo_parametros)
        call get_command_argument(2, archivo_resultados)
    endif
    
    ! Leer parámetros
    call leer_parametros(trim(archivo_parametros), params, Nx, Ny, x_range, y_range)
    
    ! Mostrar parámetros
    print *, ""
    print *, "PARAMETROS DE LA SIMULACION:"
    print *, "============================="
    print *, "Longitud de onda: ", params%wavelength*1e9, " nm"
    print *, "Rendija 1: ", params%p*1000, "x", params%q*1000, " mm"
    print *, "Rendija 2: ", params%p2*1000, "x", params%q2*1000, " mm"
    print *, "Desplazamiento rendija 2: (", params%a*1000, ", ", params%b*1000, ") mm"
    print *, "d1 = ", params%c*1000, " mm"
    print *, "d2 = ", (params%z0 - params%n - params%c)*1000, " mm"
    print *, "Resolucion: ", Nx, "x", Ny, " puntos"
    print *, "Rango observacion: ", x_range*1000, "x", y_range*1000, " mm"
    print *, ""
    
    ! Ejecutar simulación
    print *, "INICIANDO SIMULACION..."
    print *, "NOTA: Esta simulacion puede tomar varios minutos."
    print *, ""
    
    call simulate_2d_diffraction_complete(params, Nx, Ny, x_range, y_range, results)
    
    ! Imprimir resumen
    call imprimir_resultados(results)
    
    ! Guardar resultados
    call guardar_resultados(trim(archivo_resultados), results)
    
    print *, ""
    print *, "SIMULACION COMPLETADA EXITOSAMENTE"
    print *, "==================================="
    
    ! Liberar memoria
    deallocate(results%x_obs, results%y_obs)
    deallocate(results%I_fresnel, results%I_hf)
    deallocate(results%U_fresnel, results%U_hf)
    deallocate(results%I_x_fresnel, results%I_x_hf)
    
end program difraccion_main