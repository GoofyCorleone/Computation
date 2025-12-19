module system_params
    implicit none
    ! Geometric parameters
    real(8), parameter :: W = 3.0d0, H = 2.0d0, gap = 0.3d0
    real(8), parameter :: x0 = gap, x1 = W - gap, y0 = gap, y1 = H - gap
    
    ! Inlet/outlet parameters
    real(8), parameter :: lenSpecial = H / 5.0d0, lenOut = H / 5.0d0
    real(8), parameter :: centerSpecial1 = 0.65d0 * H, centerSpecial2 = 0.35d0 * H
    real(8), parameter :: centerOut1 = 0.65d0 * H, centerOut2 = 0.35d0 * H
    
    ! Physical parameters
    real(8), parameter :: dt = 0.1d0, Re = 100.0d0, Uin = 1.0d0, Tin = 1.0d0
    real(8), parameter :: alpha = 10.0d0, picardRelax = 1.0d0
    
    ! Numerical parameters
    integer, parameter :: nSteps = 200, nPicard = 5
    integer, parameter :: nx = 120, ny = 80  ! Grid resolution
    
    ! Boundary labels
    integer, parameter :: BDRY_INNER = 1, BDRY_BOTTOM = 2, BDRY_WALL = 3
    integer, parameter :: BDRY_TOP = 4, BDRY_INLET1 = 10, BDRY_INLET2 = 11
    integer, parameter :: BDRY_OUTLET1 = 20, BDRY_OUTLET2 = 21
    
contains
    
    pure logical function is_inner(x, y)
        real(8), intent(in) :: x, y
        is_inner = (x > x0 .and. x < x1 .and. y > y0 .and. y < y1)
    end function is_inner
    
    pure logical function is_inlet1(y)
        real(8), intent(in) :: y
        real(8) :: yStart1, yEnd1
        yStart1 = centerSpecial1 - lenSpecial/2.0d0
        yEnd1   = centerSpecial1 + lenSpecial/2.0d0
        is_inlet1 = (y >= yStart1 .and. y <= yEnd1)
    end function is_inlet1
    
    pure logical function is_inlet2(y)
        real(8), intent(in) :: y
        real(8) :: yStart2, yEnd2
        yStart2 = centerSpecial2 - lenSpecial/2.0d0
        yEnd2   = centerSpecial2 + lenSpecial/2.0d0
        is_inlet2 = (y >= yStart2 .and. y <= yEnd2)
    end function is_inlet2
    
    pure logical function is_outlet1(y)
        real(8), intent(in) :: y
        real(8) :: yStartOut1, yEndOut1
        yStartOut1 = centerOut1 - lenOut/2.0d0
        yEndOut1   = centerOut1 + lenOut/2.0d0
        is_outlet1 = (y >= yStartOut1 .and. y <= yEndOut1)
    end function is_outlet1
    
    pure logical function is_outlet2(y)
        real(8), intent(in) :: y
        real(8) :: yStartOut2, yEndOut2
        yStartOut2 = centerOut2 - lenOut/2.0d0
        yEndOut2   = centerOut2 + lenOut/2.0d0
        is_outlet2 = (y >= yStartOut2 .and. y <= yEndOut2)
    end function is_outlet2

end module system_params

module field_vars
    use system_params
    implicit none
    
    ! Field variables
    real(8) :: u1(0:nx+1, 0:ny+1), u2(0:nx+1, 0:ny+1), p(0:nx+1, 0:ny+1)
    real(8) :: u1old(0:nx+1, 0:ny+1), u2old(0:nx+1, 0:ny+1)
    real(8) :: Tf(0:nx+1, 0:ny+1), Ti(0:nx+1, 0:ny+1)
    real(8) :: Tfold(0:nx+1, 0:ny+1), Tiold(0:nx+1, 0:ny+1)
    real(8) :: Tcombined(0:nx+1, 0:ny+1)
    
    ! Grid
    real(8) :: x(0:nx+1), y(0:ny+1), dx, dy
    
contains
    
    subroutine initialize_fields()
        integer :: i, j
        
        dx = W / real(nx, 8)
        dy = H / real(ny, 8)
        
        ! Initialize grid
        do i = 0, nx+1
            x(i) = real(i, 8) * dx
        end do
        do j = 0, ny+1
            y(j) = real(j, 8) * dy
        end do
        
        ! Initialize fields
        u1 = 0.0d0; u2 = 0.0d0; p = 0.0d0
        u1old = 0.0d0; u2old = 0.0d0
        Tf = 0.0d0; Ti = 0.0d0
        Tfold = 0.0d0; Tiold = 0.3d0
        
    end subroutine initialize_fields

end module field_vars

module boundary_conditions
    use system_params
    use field_vars
    implicit none

contains
    
    subroutine apply_velocity_bc()
        integer :: i, j
        
        ! Apply boundary conditions for velocity
        do j = 0, ny+1
            ! Left boundary
            if (is_inlet1(y(j))) then
                u1(0,j) = Uin
                u2(0,j) = 0.0d0
            else if (is_inlet2(y(j))) then
                u1(0,j) = Uin
                u2(0,j) = 0.0d0
            else
                u1(0,j) = 0.0d0
                u2(0,j) = 0.0d0
            end if
            
            ! Right boundary
            if (is_outlet1(y(j)) .or. is_outlet2(y(j))) then
                ! Outlet - Neumann condition
                u1(nx+1,j) = u1(nx,j)
                u2(nx+1,j) = u2(nx,j)
            else
                u1(nx+1,j) = 0.0d0
                u2(nx+1,j) = 0.0d0
            end if
        end do
        
        do i = 0, nx+1
            ! Bottom boundary
            u1(i,0) = 0.0d0
            u2(i,0) = 0.0d0
            
            ! Top boundary
            u1(i,ny+1) = 0.0d0
            u2(i,ny+1) = 0.0d0
        end do
        
    end subroutine apply_velocity_bc
    
    subroutine apply_temperature_bc()
        integer :: i, j
        
        ! Apply boundary conditions for temperature
        do j = 0, ny+1
            ! Left boundary - inlets
            if (is_inlet1(y(j)) .or. is_inlet2(y(j))) then
                Tf(0,j) = Tin
                Ti(0,j) = Ti(1,j)  ! Neumann for inner domain
            else
                Tf(0,j) = Tf(1,j)  ! Neumann for walls
                Ti(0,j) = Ti(1,j)
            end if
            
            ! Right boundary - outlets (Neumann)
            Tf(nx+1,j) = Tf(nx,j)
            Ti(nx+1,j) = Ti(nx,j)
        end do
        
        do i = 0, nx+1
            ! Bottom boundary (Neumann)
            Tf(i,0) = Tf(i,1)
            Ti(i,0) = Ti(i,1)
            
            ! Top boundary (Neumann)
            Tf(i,ny+1) = Tf(i,ny)
            Ti(i,ny+1) = Ti(i,ny)
        end do
        
    end subroutine apply_temperature_bc

end module boundary_conditions

module navier_stokes_solver
    use system_params
    use field_vars
    use boundary_conditions
    implicit none
    
    private
    public :: solve_navier_stokes
    
contains
    
    subroutine solve_navier_stokes()
        integer :: i, j, k
        real(8) :: u1_temp, u2_temp
        real(8) :: conv1, conv2, diff1, diff2, press_grad1, press_grad2
        
        do k = 1, nPicard
            call apply_velocity_bc()
            
            ! Solve momentum equations with Picard iteration
            do j = 1, ny
                do i = 1, nx
                    if (is_inner(x(i), y(j))) cycle  ! Skip inner solid region
                    
                    ! Convection terms
                    conv1 = u1old(i,j) * (u1old(i+1,j) - u1old(i-1,j)) / (2.0d0*dx) + &
                           u2old(i,j) * (u1old(i,j+1) - u1old(i,j-1)) / (2.0d0*dy)
                    
                    conv2 = u1old(i,j) * (u2old(i+1,j) - u2old(i-1,j)) / (2.0d0*dx) + &
                           u2old(i,j) * (u2old(i,j+1) - u2old(i,j-1)) / (2.0d0*dy)
                    
                    ! Diffusion terms
                    diff1 = (u1old(i+1,j) - 2.0d0*u1old(i,j) + u1old(i-1,j)) / (dx*dx) + &
                           (u1old(i,j+1) - 2.0d0*u1old(i,j) + u1old(i,j-1)) / (dy*dy)
                    
                    diff2 = (u2old(i+1,j) - 2.0d0*u2old(i,j) + u2old(i-1,j)) / (dx*dx) + &
                           (u2old(i,j+1) - 2.0d0*u2old(i,j) + u2old(i,j-1)) / (dy*dy)
                    
                    ! Pressure gradient
                    press_grad1 = (p(i+1,j) - p(i-1,j)) / (2.0d0*dx)
                    press_grad2 = (p(i,j+1) - p(i,j-1)) / (2.0d0*dy)
                    
                    ! Update velocity (temporal discretization)
                    u1_temp = u1old(i,j) + dt * (-conv1 + diff1/Re - press_grad1)
                    u2_temp = u2old(i,j) + dt * (-conv2 + diff2/Re - press_grad2)
                    
                    ! Relaxation
                    u1(i,j) = (1.0d0 - picardRelax) * u1old(i,j) + picardRelax * u1_temp
                    u2(i,j) = (1.0d0 - picardRelax) * u2old(i,j) + picardRelax * u2_temp
                end do
            end do
            
            ! Solve pressure Poisson equation (simplified)
            call solve_pressure()
            
            ! Update old values
            u1old = u1
            u2old = u2
            
        end do
        
    end subroutine solve_navier_stokes
    
    subroutine solve_pressure()
        ! Simplified pressure Poisson solver
        integer :: i, j, iter
        real(8) :: div, p_new, residual
        real(8), parameter :: tolerance = 1.0d-6
        integer, parameter :: max_iter = 1000
        
        do iter = 1, max_iter
            residual = 0.0d0
            
            do j = 1, ny
                do i = 1, nx
                    if (is_inner(x(i), y(j))) cycle
                    
                    div = (u1(i+1,j) - u1(i-1,j)) / (2.0d0*dx) + &
                         (u2(i,j+1) - u2(i,j-1)) / (2.0d0*dy)
                    
                    p_new = 0.25d0 * (p(i+1,j) + p(i-1,j) + p(i,j+1) + p(i,j-1) - dx*dy*div)
                    
                    residual = residual + abs(p_new - p(i,j))
                    p(i,j) = p_new
                end do
            end do
            
            if (residual < tolerance) exit
        end do
        
    end subroutine solve_pressure

end module navier_stokes_solver

module heat_solver
    use system_params
    use field_vars
    use boundary_conditions
    implicit none
    
    private
    public :: solve_heat_equation
    
contains
    
    subroutine solve_heat_equation()
        integer :: i, j
        real(8) :: kappa_f, kappa_i, conv, diff_f, diff_i, coupling
        
        kappa_f = 1.0d0  ! Thermal conductivity for fluid
        kappa_i = 0.5d0  ! Thermal conductivity for inner region
        
        call apply_temperature_bc()
        
        do j = 1, ny
            do i = 1, nx
                ! Convection term (only in fluid region)
                conv = 0.0d0
                if (.not. is_inner(x(i), y(j))) then
                    conv = u1(i,j) * (Tfold(i+1,j) - Tfold(i-1,j)) / (2.0d0*dx) + &
                          u2(i,j) * (Tfold(i,j+1) - Tfold(i,j-1)) / (2.0d0*dy)
                end if
                
                ! Diffusion terms
                diff_f = kappa_f * ((Tfold(i+1,j) - 2.0d0*Tfold(i,j) + Tfold(i-1,j)) / (dx*dx) + &
                        (Tfold(i,j+1) - 2.0d0*Tfold(i,j) + Tfold(i,j-1)) / (dy*dy))
                
                diff_i = kappa_i * ((Tiold(i+1,j) - 2.0d0*Tiold(i,j) + Tiold(i-1,j)) / (dx*dx) + &
                        (Tiold(i,j+1) - 2.0d0*Tiold(i,j) + Tiold(i,j-1)) / (dy*dy))
                
                ! Coupling at interface
                coupling = 0.0d0
                if (on_inner_boundary(i,j)) then
                    coupling = alpha * (Tf(i,j) - Ti(i,j))
                end if
                
                ! Update temperatures
                if (is_inner(x(i), y(j))) then
                    ! Inner region
                    Ti(i,j) = Tiold(i,j) + dt * (diff_i + coupling)
                else
                    ! Fluid region
                    Tf(i,j) = Tfold(i,j) + dt * (-conv + diff_f - coupling)
                end if
            end do
        end do
        
        ! Update old values
        Tfold = Tf
        Tiold = Ti
        
        ! Combine for visualization
        do j = 0, ny+1
            do i = 0, nx+1
                if (is_inner(x(i), y(j))) then
                    Tcombined(i,j) = Ti(i,j)
                else
                    Tcombined(i,j) = Tf(i,j)
                end if
            end do
        end do
        
    end subroutine solve_heat_equation
    
    pure logical function on_inner_boundary(i, j)
        integer, intent(in) :: i, j
        real(8) :: x_val, y_val
        
        x_val = x(i)
        y_val = y(j)
        
        on_inner_boundary = .false.
        
        ! Check if this point is adjacent to inner boundary
        if (is_inner(x_val, y_val)) then
            if (.not. is_inner(x(i-1), y_val) .or. .not. is_inner(x(i+1), y_val) .or. &
                .not. is_inner(x_val, y(j-1)) .or. .not. is_inner(x_val, y(j+1))) then
                on_inner_boundary = .true.
            end if
        else
            if (is_inner(x(i-1), y_val) .or. is_inner(x(i+1), y_val) .or. &
                is_inner(x_val, y(j-1)) .or. is_inner(x_val, y(j+1))) then
                on_inner_boundary = .true.
            end if
        end if
        
    end function on_inner_boundary

end module heat_solver

program main
    use system_params
    use field_vars
    use navier_stokes_solver
    use heat_solver
    implicit none
    
    integer :: it
    character(len=50) :: filename
    
    call initialize_fields()
    
    print *, 'Starting simulation...'
    
    do it = 1, nSteps
        print *, 'Time step: ', it
        
        ! Solve Navier-Stokes equations
        call solve_navier_stokes()
        
        ! Solve heat equation
        call solve_heat_equation()
        
        ! Output results periodically
        if (mod(it, 20) == 0) then
            write(filename, '(A,I4.4,A)') 'temperature_', it, '.dat'
            call output_field(Tcombined, trim(filename))
            
            write(filename, '(A,I4.4,A)') 'velocity_', it, '.dat'
            call output_velocity(trim(filename))
        end if
    end do
    
    print *, 'Simulation completed.'
    
contains
    
    subroutine output_field(field, filename)
        real(8), intent(in) :: field(0:nx+1, 0:ny+1)
        character(len=*), intent(in) :: filename
        integer :: i, j, unit
        
        open(newunit=unit, file=filename, status='replace')
        write(unit, *) 'VARIABLES = "X", "Y", "T"'
        write(unit, *) 'ZONE I=', nx+2, ', J=', ny+2, ', F=POINT'
        
        do j = 0, ny+1
            do i = 0, nx+1
                write(unit, '(3E15.7)') x(i), y(j), field(i,j)
            end do
        end do
        
        close(unit)
    end subroutine output_field
    
    subroutine output_velocity(filename)
        character(len=*), intent(in) :: filename
        integer :: i, j, unit
        
        open(newunit=unit, file=filename, status='replace')
        write(unit, *) 'VARIABLES = "X", "Y", "U", "V"'
        write(unit, *) 'ZONE I=', nx+2, ', J=', ny+2, ', F=POINT'
        
        do j = 0, ny+1
            do i = 0, nx+1
                write(unit, '(4E15.7)') x(i), y(j), u1(i,j), u2(i,j)
            end do
        end do
        
        close(unit)
    end subroutine output_velocity

end program main