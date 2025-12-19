! schrodinger_barrier.f90
! Solución numérica de la ecuación de Schrödinger dependiente del tiempo
! con una barrera de potencial (función Heaviside) mostrando efecto túnel
! Utiliza el método de Crank-Nicolson para la evolución temporal

program schrodinger_barrier
    implicit none
    
    ! Precisión doble para todos los cálculos
    integer, parameter :: dp = selected_real_kind(15, 307)
    
    ! Constantes físicas (unidades atómicas: ħ = 1, m = 1)
    real(dp), parameter :: L = 100.0_dp          ! Longitud del dominio espacial
    integer, parameter :: N = 1000               ! Número de puntos espaciales
    real(dp), parameter :: dx = L / real(N, dp)  ! Paso espacial
    
    real(dp), parameter :: dt = 0.01_dp          ! Paso temporal
    real(dp), parameter :: t_final = 20.0_dp     ! Tiempo final de simulación
    integer, parameter :: steps = int(t_final / dt) ! Número de pasos temporales
    integer, parameter :: save_interval = 10     ! Guardar cada 'save_interval' pasos
    
    ! Parámetros de la partícula y la barrera
    real(dp), parameter :: k0 = 1.0_dp           ! Número de onda inicial
    real(dp), parameter :: x0 = L / 4.0_dp       ! Posición inicial del paquete
    real(dp), parameter :: sigma = 4.0_dp        ! Ancho del paquete de onda
    real(dp), parameter :: barrier_pos = L / 2.0_dp ! Posición de la barrera
    real(dp), parameter :: barrier_height = 0.5_dp  ! Altura de la barrera
    
    ! Variables para el cálculo
    real(dp), allocatable :: x(:)                ! Posiciones
    real(dp), allocatable :: V(:)                ! Potencial
    complex(dp), allocatable :: psi(:)           ! Función de onda
    complex(dp), allocatable :: psi_new(:)       ! Función de onda en paso t+dt
    complex(dp) :: ci = (0.0_dp, 1.0_dp)         ! Unidad imaginaria
    
    ! Variables para el método de Crank-Nicolson
    complex(dp), allocatable :: alpha(:)         ! Coeficientes para resolver sist. tridiagonal
    complex(dp), allocatable :: beta(:)          ! Coeficientes para resolver sist. tridiagonal
    real(dp) :: r                                ! Coeficiente para el esquema CN
    complex(dp), allocatable :: diagonal(:)      ! Diagonal principal
    complex(dp), allocatable :: off_diag(:)      ! Elementos fuera de la diagonal
    complex(dp), allocatable :: b(:)             ! Lado derecho del sistema
    
    ! Variables para salida de datos
    integer :: i, t_i, save_count
    character(len=100) :: filename
    integer :: fileunit = 10
    
    ! Inicializar arrays
    allocate(x(N))
    allocate(V(N))
    allocate(psi(N))
    allocate(psi_new(N))
    allocate(alpha(N))
    allocate(beta(N))
    allocate(diagonal(N))
    allocate(off_diag(N-1))
    allocate(b(N))
    
    ! Crear la malla espacial
    do i = 1, N
        x(i) = (i-1) * dx
    end do
    
    ! Definir el potencial (barrera de Heaviside)
    V = 0.0_dp
    do i = 1, N
        if (x(i) >= barrier_pos) then
            V(i) = barrier_height
        end if
    end do
    
    ! Crear el paquete de onda inicial (gaussiano con momento k0)
    do i = 1, N
        psi(i) = exp(-(x(i) - x0)**2 / (2.0_dp * sigma**2)) * exp(ci * k0 * x(i))
    end do
    
    ! Normalizar la función de onda
    psi = psi / sqrt(sum(abs(psi)**2) * dx)
    
    ! Coeficiente para el esquema de Crank-Nicolson
    r = dt / (2.0_dp * dx**2)
    
    ! Preparar archivo para guardar datos de posición y potencial
    open(unit=fileunit, file='grid_data.dat', status='replace')
    write(fileunit, *) '# x V'
    do i = 1, N
        write(fileunit, *) x(i), V(i)
    end do
    close(fileunit)
    
    ! Crear directorio para los archivos de salida (el sistema operativo debe soportarlo)
    call system('mkdir -p wavedata')
    
    ! Guardar el estado inicial
    save_count = 0
    write(filename, '(a,i5.5,a)') 'wavedata/psi_', save_count, '.dat'
    open(unit=fileunit, file=trim(filename), status='replace')
    write(fileunit, *) '# t = ', 0.0_dp
    write(fileunit, *) '# x re(psi) im(psi) |psi|^2'
    do i = 1, N
        write(fileunit, *) x(i), real(psi(i)), aimag(psi(i)), abs(psi(i))**2
    end do
    close(fileunit)
    
    ! Evolución temporal usando el método de Crank-Nicolson
    do t_i = 1, steps
        ! Construir las matrices tridiagonales para el sistema lineal
        diagonal = 1.0_dp + 2.0_dp*ci*r + ci*dt*V/2.0_dp
        off_diag = -ci*r
        
        ! Vector del lado derecho
        do i = 2, N-1
            b(i) = ci*r*psi(i-1) + (1.0_dp - 2.0_dp*ci*r - ci*dt*V(i)/2.0_dp)*psi(i) + ci*r*psi(i+1)
        end do
        b(1) = (1.0_dp - 2.0_dp*ci*r - ci*dt*V(1)/2.0_dp)*psi(1) + ci*r*psi(2)
        b(N) = ci*r*psi(N-1) + (1.0_dp - 2.0_dp*ci*r - ci*dt*V(N)/2.0_dp)*psi(N)
        
        ! Resolver el sistema tridiagonal (algoritmo de Thomas)
        alpha(1) = diagonal(1)
        beta(1) = b(1) / alpha(1)
        
        do i = 2, N
            alpha(i) = diagonal(i) - off_diag(i-1)*off_diag(i-1)/alpha(i-1)
            beta(i) = (b(i) - off_diag(i-1)*beta(i-1)) / alpha(i)
        end do
        
        ! Sustitución hacia atrás
        psi_new(N) = beta(N)
        do i = N-1, 1, -1
            psi_new(i) = beta(i) - off_diag(i)*psi_new(i+1)/alpha(i)
        end do
        
        ! Actualizar la función de onda
        psi = psi_new
        
        ! Guardar los resultados periódicamente
        if (mod(t_i, save_interval) == 0) then
            save_count = save_count + 1
            write(filename, '(a,i5.5,a)') 'wavedata/psi_', save_count, '.dat'
            open(unit=fileunit, file=trim(filename), status='replace')
            write(fileunit, *) '# t = ', t_i*dt
            write(fileunit, *) '# x re(psi) im(psi) |psi|^2'
            do i = 1, N
                write(fileunit, *) x(i), real(psi(i)), aimag(psi(i)), abs(psi(i))**2
            end do
            close(fileunit)
        end if
    end do
    
    ! Guardar el número total de archivos de datos generados
    open(unit=fileunit, file='filecount.dat', status='replace')
    write(fileunit, *) save_count
    close(fileunit)
    
    ! Liberar memoria
    deallocate(x, V, psi, psi_new, alpha, beta, diagonal, off_diag, b)
    
    print *, 'Simulación completada. Archivos de salida guardados en el directorio wavedata/'
    
end program schrodinger_barrier