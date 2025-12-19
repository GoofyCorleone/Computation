program main
    use Global_parameters
    use funcmod
    implicit none

    REAL :: start , finish
    CALL CPU_TIME(start)
    pii = 4.0d0*ATAN(1.0d0)
    ox = 4 , oy = 7

    print *, 'Number of colocation points'
    read(*,*) N
    print *

    print *, 'Number of grid points'
    read(*,*) Nx
    print *
    Ny = Nx

    CALL memory
    call Vecs
    CALL saving
    CALL CPU_TIME(finish)
    print*, 'Tiempo de ejecución = ' , finish , '[s]'

end program main