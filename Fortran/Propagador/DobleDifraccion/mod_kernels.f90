!===============================================================================
! MODULO: mod_kernels
! Propósito: Implementa los kernels de Fresnel y Huygens-Fresnel
!===============================================================================
module mod_kernels
    use mod_constantes
    implicit none
    
    private
    public :: fresnel_kernel_2d, huygens_fresnel_kernel_2d, rect_function_2d
    
contains
    
    !-----------------------------------------------------------------------
    ! Kernel de Fresnel 2D (aproximación paraxial)
    !-----------------------------------------------------------------------
    function fresnel_kernel_2d(x0, y0, x, y, d, k) result(kernel)
        real(dp), intent(in) :: x0, y0, x, y, d, k
        complex(dp) :: kernel
        
        real(dp) :: r2
        
        if (abs(d) < 1.0e-12_dp) then
            kernel = (0.0_dp, 0.0_dp)
            return
        endif
        
        r2 = (x0 - x)**2 + (y0 - y)**2
        kernel = exp(ii * k * r2 / (2.0_dp * d))
        
    end function fresnel_kernel_2d
    
    !-----------------------------------------------------------------------
    ! Kernel de Huygens-Fresnel 2D (versión exacta estabilizada)
    !-----------------------------------------------------------------------
    function huygens_fresnel_kernel_2d(x0, y0, x, y, d, k) result(kernel)
        real(dp), intent(in) :: x0, y0, x, y, d, k
        complex(dp) :: kernel
        
        real(dp) :: dx, dy, r, cos_theta
        
        if (abs(d) < 1.0e-12_dp) then
            kernel = (0.0_dp, 0.0_dp)
            return
        endif
        
        dx = x0 - x
        dy = y0 - y
        r = sqrt(dx**2 + dy**2 + d**2)
        
        if (r < 1.0e-10_dp) then
            kernel = (0.0_dp, 0.0_dp)
            return
        endif
        
        ! Factor de oblicuidad
        cos_theta = d / r
        
        ! Kernel completo
        kernel = (cos_theta / r) * exp(ii * k * r)
        
    end function huygens_fresnel_kernel_2d
    
    !-----------------------------------------------------------------------
    ! Función rectangular 2D (apertura)
    !-----------------------------------------------------------------------
    function rect_function_2d(x, y, center_x, center_y, width_x, width_y) result(val)
        real(dp), intent(in) :: x, y, center_x, center_y, width_x, width_y
        real(dp) :: val
        
        if (abs(x - center_x) <= width_x/2.0_dp .and. &
            abs(y - center_y) <= width_y/2.0_dp) then
            val = 1.0_dp
        else
            val = 0.0_dp
        endif
        
    end function rect_function_2d
    
end module mod_kernels