!===============================================================================
! MODULO: mod_simulacion
! Propósito: Ejecuta la simulación completa para ambos métodos
!===============================================================================
module mod_simulacion
    use mod_constantes
    use mod_campos
    implicit none
    
    private
    public :: simulate_2d_diffraction_complete
    
contains
    
    !-----------------------------------------------------------------------
    ! Simulación 2D completa para ambos métodos
    !-----------------------------------------------------------------------
    subroutine simulate_2d_diffraction_complete(params, Nx, Ny, x_range, y_range, results)
        type(parametros_sistema), intent(in) :: params
        integer, intent(in) :: Nx, Ny
        real(dp), intent(in) :: x_range, y_range
        type(resultados_simulacion), intent(out) :: results
        
        integer :: i, j, center_y_idx
        real(dp) :: start_time, end_time, max_val
        complex(dp) :: U
        type(config_integracion) :: config
        
        ! Configuración de integración
        config%max_depth = 15
        config%max_depth_2d = 6
        config%tol_1d = 1.0e-6_dp
        config%tol_2d = 1.0e-5_dp
        config%tol_integrand = 1.0e-5_dp
        
        ! Inicializar arrays
        allocate(results%x_obs(Nx), results%y_obs(Ny))
        allocate(results%I_fresnel(Nx, Ny), results%I_hf(Nx, Ny))
        allocate(results%U_fresnel(Nx, Ny), results%U_hf(Nx, Ny))
        allocate(results%I_x_fresnel(Nx), results%I_x_hf(Nx))
        
        results%Nx = Nx
        results%Ny = Ny
        results%params = params
        
        ! Crear mallas de observación
        do i = 1, Nx
            results%x_obs(i) = -x_range/2.0_dp + (i-1) * x_range/(Nx-1)
        end do
        
        do j = 1, Ny
            results%y_obs(j) = -y_range/2.0_dp + (j-1) * y_range/(Ny-1)
        end do
        
        ! Método Fresnel
        print *, "Calculando patrón de Fresnel (2D completo)..."
        call cpu_time(start_time)
        
        do i = 1, Nx
            if (mod(i, 10) == 0) then
                print *, "  Progreso Fresnel: ", i, "/", Nx, " puntos en x"
            endif
            
            do j = 1, Ny
                U = calculate_field_observation_2d(results%x_obs(i), results%y_obs(j), &
                                                  params, 'fresnel', config)
                results%U_fresnel(i, j) = U
                results%I_fresnel(i, j) = abs(U)**2
            end do
        end do
        
        call cpu_time(end_time)
        results%tiempo_fresnel = end_time - start_time
        print *, "Tiempo cálculo Fresnel: ", results%tiempo_fresnel, " s"
        
        ! Método Huygens-Fresnel
        print *, ""
        print *, "Calculando patrón de Huygens-Fresnel (2D completo)..."
        call cpu_time(start_time)
        
        do i = 1, Nx
            if (mod(i, 10) == 0) then
                print *, "  Progreso Huygens-Fresnel: ", i, "/", Nx, " puntos en x"
            endif
            
            do j = 1, Ny
                U = calculate_field_observation_2d(results%x_obs(i), results%y_obs(j), &
                                                  params, 'huygens_fresnel', config)
                results%U_hf(i, j) = U
                results%I_hf(i, j) = abs(U)**2
            end do
        end do
        
        call cpu_time(end_time)
        results%tiempo_hf = end_time - start_time
        print *, "Tiempo cálculo Huygens-Fresnel: ", results%tiempo_hf, " s"
        
        ! Normalizar intensidades
        max_val = maxval(results%I_fresnel)
        if (max_val > 0.0_dp) then
            results%I_fresnel = results%I_fresnel / max_val
        endif
        
        max_val = maxval(results%I_hf)
        if (max_val > 0.0_dp) then
            results%I_hf = results%I_hf / max_val
        endif
        
        ! Perfiles en x (y=0)
        center_y_idx = Ny/2 + 1
        results%I_x_fresnel = results%I_fresnel(:, center_y_idx)
        results%I_x_hf = results%I_hf(:, center_y_idx)
        
    end subroutine simulate_2d_diffraction_complete
    
end module mod_simulacion