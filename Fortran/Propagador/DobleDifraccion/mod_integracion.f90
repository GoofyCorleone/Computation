!===============================================================================
! MODULO: mod_integracion
! Propósito: Implementa integración adaptativa de Simpson 1D y 2D
!===============================================================================
module mod_integracion
    use mod_constantes
    implicit none
    
    ! Tipo para función 1D
    abstract interface
        function func_1d(x) result(f)
            import :: dp
            real(dp), intent(in) :: x
            complex(dp) :: f
        end function func_1d
    end interface
    
    ! Tipo para función 2D
    abstract interface
        function func_2d(x, y) result(f)
            import :: dp
            real(dp), intent(in) :: x, y
            complex(dp) :: f
        end function func_2d
    end interface
    
    ! Variables globales para comunicación con funciones internas
    real(dp), save :: current_x
    procedure(func_2d), pointer, save :: current_f => null()
    
contains
    
    !-----------------------------------------------------------------------
    ! Integración adaptativa de Simpson 1D (recursiva)
    !-----------------------------------------------------------------------
    recursive function simpson_adaptive_1d(f, a, b, tol, max_depth, depth) result(integral)
        procedure(func_1d) :: f
        real(dp), intent(in) :: a, b, tol
        integer, intent(in) :: max_depth
        integer, intent(in), optional :: depth
        complex(dp) :: integral
        
        real(dp) :: h, mid, left, right
        complex(dp) :: f_a, f_b, f_mid, f_left_mid, f_right_mid
        complex(dp) :: whole, left_half, right_half, total
        integer :: current_depth
        
        ! Manejar profundidad actual
        if (present(depth)) then
            current_depth = depth
        else
            current_depth = 0
        endif
        
        ! Evaluar en los puntos
        f_a = f(a)
        f_b = f(b)
        mid = (a + b) / 2.0_dp
        f_mid = f(mid)
        
        ! Regla de Simpson simple
        h = b - a
        whole = h * (f_a + 4.0_dp * f_mid + f_b) / 6.0_dp
        
        ! Si alcanzamos profundidad máxima, retornar
        if (current_depth >= max_depth) then
            integral = whole
            return
        endif
        
        ! Evaluar puntos medios
        left = (a + mid) / 2.0_dp
        right = (mid + b) / 2.0_dp
        f_left_mid = f(left)
        f_right_mid = f(right)
        
        ! Calcular mitades
        left_half = (mid - a) * (f_a + 4.0_dp * f_left_mid + f_mid) / 6.0_dp
        right_half = (b - mid) * (f_mid + 4.0_dp * f_right_mid + f_b) / 6.0_dp
        total = left_half + right_half
        
        ! Criterio de convergencia
        if (abs(total - whole) <= 15.0_dp * tol) then
            integral = total + (total - whole) / 15.0_dp
            return
        endif
        
        ! Recursión
        integral = simpson_adaptive_1d(f, a, mid, tol, max_depth, current_depth + 1) + &
                   simpson_adaptive_1d(f, mid, b, tol, max_depth, current_depth + 1)
        
    end function simpson_adaptive_1d
    
    !-----------------------------------------------------------------------
    ! Función auxiliar para integración en y (necesaria para comunicación)
    !-----------------------------------------------------------------------
    function y_func_wrapper(y) result(val)
        real(dp), intent(in) :: y
        complex(dp) :: val
        
        ! Usa current_x y current_f que fueron establecidos por la función contenedora
        val = current_f(current_x, y)
    end function y_func_wrapper
    
    !-----------------------------------------------------------------------
    ! Función para integrar en y para un x fijo
    !-----------------------------------------------------------------------
    function integrate_y_for_fixed_x(x, f, ay, by, tol, max_depth) result(integral_y)
        real(dp), intent(in) :: x
        procedure(func_2d) :: f
        real(dp), intent(in) :: ay, by, tol
        integer, intent(in) :: max_depth
        complex(dp) :: integral_y
        
        ! Guardar x y f en variables globales para y_func_wrapper
        current_x = x
        current_f => f
        
        ! Integrar en y
        integral_y = simpson_adaptive_1d(y_func_wrapper, ay, by, tol, max_depth)
    end function integrate_y_for_fixed_x
    
    !-----------------------------------------------------------------------
    ! Integración 2D adaptativa (anidada)
    !-----------------------------------------------------------------------
    function simpson_adaptive_2d(f, ax, bx, ay, by, tol, max_depth) result(integral)
        procedure(func_2d) :: f
        real(dp), intent(in) :: ax, bx, ay, by, tol
        integer, intent(in) :: max_depth
        complex(dp) :: integral
        
        ! Función interna para integrar en x
        integral = simpson_adaptive_1d(integrate_x_wrapper, ax, bx, tol, max_depth)
        
    contains
    
        function integrate_x_wrapper(x) result(f_x)
            real(dp), intent(in) :: x
            complex(dp) :: f_x
            
            ! Para cada x, integrar en y
            f_x = integrate_y_for_fixed_x(x, f, ay, by, tol/10.0_dp, max_depth-2)
        end function integrate_x_wrapper
        
    end function simpson_adaptive_2d
    
end module mod_integracion