module arrays
    implicit none

    !! =============Para la función en X===============================================
    REAL(kind=8), ALLOCATABLE, DIMENSION(:,:) :: Ax  ! matriz para la función en x
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: x    ! puntos de colocación en x
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: bx    ! vector fuente de la función en x
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: aax   ! vector solucion, coeficients x
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: XX   ! Vector de malla en x
    !! =============Para la función en y===============================================
    REAL(kind=8), ALLOCATABLE, DIMENSION(:,:) :: Ay  ! matriz para la función en y
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: y    ! puntos de colocación en y
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: by    ! vector fuente de la función en y
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: aay   ! vector solucion, coeficients y
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: YY   ! Vector de malla en y
    INTEGER, ALLOCATABLE, DIMENSION(:) :: indx
    REAL(kind=8), ALLOCATABLE, DIMENSION(:) :: error

end module arrays