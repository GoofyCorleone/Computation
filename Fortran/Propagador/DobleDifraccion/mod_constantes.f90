!===============================================================================
! MODULO: mod_constantes
! Propósito: Define constantes y tipos de datos para la simulación
!===============================================================================
module mod_constantes
    implicit none
    
    ! Precisión numérica
    integer, parameter :: dp = kind(1.0d0)
    real(dp), parameter :: pi = 3.14159265358979323846_dp
    complex(dp), parameter :: ii = (0.0_dp, 1.0_dp)
    
    ! Constantes físicas
    real(dp), parameter :: wavelength_def = 632.8e-9_dp  ! Longitud de onda por defecto (He-Ne)
    
    ! Parámetros del sistema
    type :: parametros_sistema
        real(dp) :: p      ! Ancho primera rendija (x)
        real(dp) :: q      ! Alto primera rendija (y)
        real(dp) :: p2     ! Ancho segunda rendija (x)
        real(dp) :: q2     ! Alto segunda rendija (y)
        real(dp) :: a      ! Desplazamiento x segunda rendija
        real(dp) :: b      ! Desplazamiento y segunda rendija
        real(dp) :: n      ! Distancia a primera rendija
        real(dp) :: c      ! Separación entre rendijas
        real(dp) :: z0     ! Distancia al plano de observación
        real(dp) :: wavelength  ! Longitud de onda
    end type parametros_sistema
    
    ! Configuración de integración
    type :: config_integracion
        integer :: max_depth = 15
        integer :: max_depth_2d = 8
        real(dp) :: tol_1d = 1.0e-6_dp
        real(dp) :: tol_2d = 1.0e-4_dp
        real(dp) :: tol_integrand = 1.0e-5_dp
    end type config_integracion
    
    ! Resultados de simulación
    type :: resultados_simulacion
        integer :: Nx, Ny
        real(dp), allocatable :: x_obs(:), y_obs(:)
        real(dp), allocatable :: I_fresnel(:,:), I_hf(:,:)
        real(dp), allocatable :: I_x_fresnel(:), I_x_hf(:)
        complex(dp), allocatable :: U_fresnel(:,:), U_hf(:,:)
        type(parametros_sistema) :: params
        real(dp) :: tiempo_fresnel, tiempo_hf
    end type resultados_simulacion
    
contains
    
    ! Inicializar parámetros por defecto
    subroutine init_parametros_defecto(params)
        type(parametros_sistema), intent(out) :: params
        
        params%p = 0.2e-3_dp
        params%q = 0.2e-3_dp
        params%p2 = 0.15e-3_dp
        params%q2 = 0.1e-3_dp
        params%a = 0.5e-3_dp
        params%b = 0.3e-3_dp
        params%n = 20.0e-3_dp
        params%c = 5.0e-3_dp
        params%z0 = 100.0e-3_dp
        params%wavelength = 632.8e-9_dp
    end subroutine init_parametros_defecto
    
end module mod_constantes