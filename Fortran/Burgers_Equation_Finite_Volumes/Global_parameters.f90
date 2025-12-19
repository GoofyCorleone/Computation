module Global_parameters
    implicit none

    !!  Grid data
    REAL(KIND=8) :: xmin
    REAL(KIND=8) :: xmax
    REAL(KIND=8) :: dx
    REAL(KIND=8) :: dt
    REAL(KIND=8) :: t

    INTEGER :: Nx,Nt
    INTEGER :: res_num
    INTEGER :: every_1D
    INTEGER :: Num 

    !! Wave data
    REAL(KIND=8) :: amp
    REAL(KIND=8) :: x0
    REAL(KIND=8) :: sigma

    !! Initial and boundary conditions and the integrator.
    CHARACTER(LEN = 20) :: integrator
    CHARACTER(LEN = 20) :: boundary
    CHARACTER(LEN = 20) :: IC

end module Global_parameters