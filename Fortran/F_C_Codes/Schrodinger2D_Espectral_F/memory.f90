subroutine memory
    use arrays
    use Global_parameters
    implicit none

    ALLOCATE(Ax(1:N+1,1:N+1))
    ALLOCATE(Ay(1:N+1,1:N+1))
    ALLOCATE(x(0:N))
    ALLOCATE(y(0:N))
    ALLOCATE(b(1:N+1))
    ALLOCATE(indx(1:N+1))
    ALLOCATE(aa(0:N))
    ALLOCATE(XX(0:Nx))
    ALLOCATE(YY(0:Ny))
    ALLOCATE(error(0:Nx))

end subroutine memory