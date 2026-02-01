!===============================================================================
! MODULO: mod_io
! Propósito: Manejo de entrada/salida de datos
!===============================================================================
module mod_io
    use mod_constantes
    implicit none
    
    private
    public :: leer_parametros, guardar_resultados, imprimir_resultados
    
contains
    
    !-----------------------------------------------------------------------
    ! Leer parámetros desde archivo
    !-----------------------------------------------------------------------
    subroutine leer_parametros(nombre_archivo, params, Nx, Ny, x_range, y_range)
        character(len=*), intent(in) :: nombre_archivo
        type(parametros_sistema), intent(out) :: params
        integer, intent(out) :: Nx, Ny
        real(dp), intent(out) :: x_range, y_range
        
        integer :: iunit, ierr
        
        open(newunit=iunit, file=nombre_archivo, status='old', action='read', iostat=ierr)
        if (ierr /= 0) then
            print *, "Error: No se pudo abrir el archivo ", trim(nombre_archivo)
            print *, "Usando parámetros por defecto"
            call init_parametros_defecto(params)
            Nx = 100
            Ny = 100
            x_range = 8.0e-3_dp
            y_range = 8.0e-3_dp
            return
        endif
        
        ! Leer parámetros
        read(iunit, *) params%p
        read(iunit, *) params%q
        read(iunit, *) params%p2
        read(iunit, *) params%q2
        read(iunit, *) params%a
        read(iunit, *) params%b
        read(iunit, *) params%n
        read(iunit, *) params%c
        read(iunit, *) params%z0
        read(iunit, *) params%wavelength
        read(iunit, *) Nx
        read(iunit, *) Ny
        read(iunit, *) x_range
        read(iunit, *) y_range
        
        close(iunit)
        
        print *, "Parámetros leídos desde: ", trim(nombre_archivo)
        
    end subroutine leer_parametros
    
    !-----------------------------------------------------------------------
    ! Guardar resultados en archivo binario (eficiente)
    !-----------------------------------------------------------------------
    subroutine guardar_resultados(nombre_archivo, results)
        character(len=*), intent(in) :: nombre_archivo
        type(resultados_simulacion), intent(in) :: results
        
        integer :: iunit
        character(len=100) :: nombre_bin
        
        ! Guardar datos binarios
        nombre_bin = trim(nombre_archivo)//".bin"
        open(newunit=iunit, file=nombre_bin, form='unformatted', status='replace')
        
        ! Guardar dimensiones
        write(iunit) results%Nx, results%Ny
        
        ! Guardar arrays
        write(iunit) results%x_obs
        write(iunit) results%y_obs
        write(iunit) results%I_fresnel
        write(iunit) results%I_hf
        write(iunit) results%I_x_fresnel
        write(iunit) results%I_x_hf
        write(iunit) results%U_fresnel
        write(iunit) results%U_hf
        
        ! Guardar parámetros
        write(iunit) results%params%p
        write(iunit) results%params%q
        write(iunit) results%params%p2
        write(iunit) results%params%q2
        write(iunit) results%params%a
        write(iunit) results%params%b
        write(iunit) results%params%n
        write(iunit) results%params%c
        write(iunit) results%params%z0
        write(iunit) results%params%wavelength
        
        ! Guardar tiempos
        write(iunit) results%tiempo_fresnel
        write(iunit) results%tiempo_hf
        
        close(iunit)
        
        ! Guardar también en formato texto para verificación
        call guardar_resultados_texto(trim(nombre_archivo)//".txt", results)
        
        print *, "Resultados guardados en: ", trim(nombre_bin)
        print *, " y en: ", trim(nombre_archivo)//".txt"
        
    end subroutine guardar_resultados
    
    !-----------------------------------------------------------------------
    ! Guardar resultados en formato texto legible
    !-----------------------------------------------------------------------
    subroutine guardar_resultados_texto(nombre_archivo, results)
        character(len=*), intent(in) :: nombre_archivo
        type(resultados_simulacion), intent(in) :: results
        
        integer :: iunit, i, j
        
        open(newunit=iunit, file=nombre_archivo, status='replace')
        
        ! Encabezado
        write(iunit, '(A)') "=========================================="
        write(iunit, '(A)') "RESULTADOS SIMULACION DIFRACCION 2D"
        write(iunit, '(A)') "=========================================="
        write(iunit, *)
        
        ! Parámetros
        write(iunit, '(A)') "PARAMETROS DEL SISTEMA:"
        write(iunit, '(A, ES12.4, A)') "λ = ", results%params%wavelength*1e9, " nm"
        write(iunit, '(A, 2F8.2, A)') "Rendija 1: ", results%params%p*1000, results%params%q*1000, " mm"
        write(iunit, '(A, 2F8.2, A)') "Rendija 2: ", results%params%p2*1000, results%params%q2*1000, " mm"
        write(iunit, '(A, 2F8.2, A)') "Desplazamiento: ", results%params%a*1000, results%params%b*1000, " mm"
        write(iunit, '(A, F8.2, A)') "d1 = ", results%params%c*1000, " mm"
        write(iunit, '(A, F8.2, A)') "d2 = ", (results%params%z0 - results%params%n - results%params%c)*1000, " mm"
        write(iunit, *)
        
        ! Tiempos
        write(iunit, '(A)') "TIEMPOS DE CALCULO:"
        write(iunit, '(A, F8.2, A)') "Fresnel: ", results%tiempo_fresnel, " s"
        write(iunit, '(A, F8.2, A)') "Huygens-Fresnel: ", results%tiempo_hf, " s"
        write(iunit, *)
        
        ! Dimensiones
        write(iunit, '(A, 2I6)') "Dimensiones: Nx, Ny = ", results%Nx, results%Ny
        write(iunit, *)
        
        ! Perfiles en x (primeros y últimos 10 puntos)
        write(iunit, '(A)') "PERFILES EN X (y=0) - Primeros 10 puntos:"
        write(iunit, '(A15, 2A20)') "x (mm)", "I_Fresnel", "I_HF"
        do i = 1, min(10, results%Nx)
            write(iunit, '(F15.6, 2F20.10)') results%x_obs(i)*1000, &
                                             results%I_x_fresnel(i), &
                                             results%I_x_hf(i)
        end do
        
        write(iunit, *)
        write(iunit, '(A)') "PERFILES EN X (y=0) - Ultimos 10 puntos:"
        write(iunit, '(A15, 2A20)') "x (mm)", "I_Fresnel", "I_HF"
        do i = max(1, results%Nx-9), results%Nx
            write(iunit, '(F15.6, 2F20.10)') results%x_obs(i)*1000, &
                                             results%I_x_fresnel(i), &
                                             results%I_x_hf(i)
        end do
        
        close(iunit)
        
    end subroutine guardar_resultados_texto
    
    !-----------------------------------------------------------------------
    ! Imprimir resultados en pantalla
    !-----------------------------------------------------------------------
    subroutine imprimir_resultados(results)
        type(resultados_simulacion), intent(in) :: results
        
        real(dp) :: diff_prom, diff_max, d_total, Fresnel_number
        
        print *, ""
        print *, "=================================================="
        print *, "RESUMEN DE RESULTADOS"
        print *, "=================================================="
        print *, "Maxima intensidad Fresnel: ", maxval(results%I_fresnel)
        print *, "Maxima intensidad Huygens-Fresnel: ", maxval(results%I_hf)
        
        ! Calcular diferencia
        diff_prom = sum(abs(results%I_fresnel - results%I_hf)) / (results%Nx * results%Ny)
        diff_max = maxval(abs(results%I_fresnel - results%I_hf))
        
        print *, "Diferencia promedio: ", diff_prom
        print *, "Diferencia maxima: ", diff_max
        
        ! Numero de Fresnel
        d_total = results%params%z0 - results%params%n
        Fresnel_number = (results%params%p**2) / (results%params%wavelength * d_total)
        
        print *, ""
        print *, "Numero de Fresnel: ", Fresnel_number
        if (Fresnel_number > 1.0_dp) then
            print *, "Region: Difraccion de Fresnel (campo cercano)"
        else
            print *, "Region: Difraccion de Fraunhofer (campo lejano)"
        endif
        
    end subroutine imprimir_resultados
    
end module mod_io