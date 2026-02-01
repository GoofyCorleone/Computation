!===============================================================================
! MODULO: mod_campos
! Propósito: Calcula campos en rendijas y plano de observación
!===============================================================================
module mod_campos
    use mod_constantes
    use mod_integracion
    use mod_kernels
    implicit none
    
    private
    
    ! Variables para comunicación con funciones de integrando
    real(dp) :: x2_save, y2_save, a_save, b_save
    real(dp) :: p_save, q_save, p2_save, q2_save
    real(dp) :: c_save, d2_save, k_save
    real(dp) :: x0_save, y0_save
    type(parametros_sistema) :: params_save
    type(config_integracion) :: config_save
    character(len=20) :: method_save = ''
    
    public :: calculate_field_at_slit2_2d, calculate_field_observation_2d
    
contains
    
    !-----------------------------------------------------------------------
    ! Integrando para el campo en la segunda rendija
    !-----------------------------------------------------------------------
    function integrand_slit2(x1, y1) result(val)
        real(dp), intent(in) :: x1, y1
        complex(dp) :: val
        
        real(dp) :: transmission
        complex(dp) :: kernel_val
        
        ! Transmitancia de la primera rendija
        transmission = rect_function_2d(x1, y1, 0.0_dp, 0.0_dp, p_save, q_save)
        
        if (abs(transmission) < 1.0e-10_dp) then
            val = (0.0_dp, 0.0_dp)
            return
        endif
        
        ! Kernel según método
        if (trim(method_save) == 'fresnel') then
            kernel_val = fresnel_kernel_2d(x2_save, y2_save, x1, y1, c_save, k_save)
        else
            kernel_val = huygens_fresnel_kernel_2d(x2_save, y2_save, x1, y1, c_save, k_save)
        endif
        
        val = transmission * kernel_val
        
    end function integrand_slit2
    
    !-----------------------------------------------------------------------
    ! Campo en la segunda rendija debido a la primera
    !-----------------------------------------------------------------------
    function calculate_field_at_slit2_2d(x2, y2, params, method, config) result(field)
        real(dp), intent(in) :: x2, y2
        type(parametros_sistema), intent(in) :: params
        character(len=*), intent(in) :: method
        type(config_integracion), intent(in) :: config
        complex(dp) :: field
        
        real(dp) :: p, q, c, k
        complex(dp) :: prefactor, integral
        
        p = params%p
        q = params%q
        c = params%c
        k = 2.0_dp * pi / params%wavelength
        
        ! Guardar variables para la función integrand
        x2_save = x2
        y2_save = y2
        p_save = p
        q_save = q
        c_save = c
        k_save = k
        method_save = method
        
        ! Prefactor según método
        if (trim(method) == 'fresnel') then
            prefactor = exp(ii * k * c) / (ii * params%wavelength * c)
        else  ! huygens_fresnel
            prefactor = 1.0_dp / (ii * params%wavelength)
        endif
        
        ! Calcular integral
        integral = simpson_adaptive_2d(integrand_slit2, &
                                      -p/2.0_dp, p/2.0_dp, &
                                      -q/2.0_dp, q/2.0_dp, &
                                      config%tol_integrand, config%max_depth_2d)
        
        field = prefactor * integral
        
    end function calculate_field_at_slit2_2d
    
    !-----------------------------------------------------------------------
    ! Integrando para el campo en el plano de observación
    !-----------------------------------------------------------------------
    function integrand_observation(x2, y2) result(val)
        real(dp), intent(in) :: x2, y2
        complex(dp) :: val
        
        real(dp) :: transmission
        complex(dp) :: field_slit2, kernel_val
        
        ! Transmitancia de la segunda rendija
        transmission = rect_function_2d(x2, y2, a_save, b_save, p2_save, q2_save)
        
        if (abs(transmission) < 1.0e-10_dp) then
            val = (0.0_dp, 0.0_dp)
            return
        endif
        
        ! Campo en la segunda rendija
        field_slit2 = calculate_field_at_slit2_2d(x2, y2, params_save, method_save, &
                                                  config_save)
        
        ! Kernel según método
        if (trim(method_save) == 'fresnel') then
            kernel_val = fresnel_kernel_2d(x0_save, y0_save, x2, y2, d2_save, k_save)
        else
            kernel_val = huygens_fresnel_kernel_2d(x0_save, y0_save, x2, y2, d2_save, k_save)
        endif
        
        val = field_slit2 * transmission * kernel_val
        
    end function integrand_observation
    
    !-----------------------------------------------------------------------
    ! Campo en el plano de observación
    !-----------------------------------------------------------------------
    function calculate_field_observation_2d(x0, y0, params, method, config) result(field)
        real(dp), intent(in) :: x0, y0
        type(parametros_sistema), intent(in) :: params
        character(len=*), intent(in) :: method
        type(config_integracion), intent(in) :: config
        complex(dp) :: field
        
        real(dp) :: p2, q2, a, b, n, c, z0, d2, k
        complex(dp) :: prefactor, integral
        
        p2 = params%p2
        q2 = params%q2
        a = params%a
        b = params%b
        n = params%n
        c = params%c
        z0 = params%z0
        d2 = z0 - (n + c)
        k = 2.0_dp * pi / params%wavelength
        
        ! Guardar variables para la función integrand
        x0_save = x0
        y0_save = y0
        p2_save = p2
        q2_save = q2
        a_save = a
        b_save = b
        d2_save = d2
        k_save = k
        params_save = params  ! Hacemos una copia de los parámetros
        method_save = method
        config_save = config
        
        ! Prefactor según método
        if (trim(method) == 'fresnel') then
            prefactor = exp(ii * k * d2) / (ii * params%wavelength * d2)
        else  ! huygens_fresnel
            prefactor = 1.0_dp / (ii * params%wavelength)
        endif
        
        ! Calcular integral
        integral = simpson_adaptive_2d(integrand_observation, &
                                      a - p2/2.0_dp, a + p2/2.0_dp, &
                                      b - q2/2.0_dp, b + q2/2.0_dp, &
                                      config%tol_integrand, config%max_depth_2d)
        
        field = prefactor * integral
        
    end function calculate_field_observation_2d
    
end module mod_campos