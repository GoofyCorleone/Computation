subroutine Vecs
    use arrays
    use Global_parameters
    use funcmod
    implicit none

    ! ================================================= !
    !   Puntos de colocación 
    xmax = 1.0d0 ; xmin = 0.0d0
    ymax = 1.0d0 ; ymin = 0.0d0
    
    do i=0,N
        x(i) = COS(pii*i/N)
        y(i) = COS(pii*i/N)
    end do

    ! ================================================= !
    !   Defining the  x_matrix.
    do i=0,N
        do l=0,N
            if (i.eq.0.or.i.eq.N) then
                Ax(i+1,l+1) = cheby(l,x(i))
            else
                Ax(i+1,l+1) = ddcheby(l,x(i))
            end if
        end do
    end do

    ! ================================================= !
    !   Defining the  y_matrix.
    do i=0,N
        do l=0,N
            if (i.eq.0.or.i.eq.N) then
                Ay(i+1,l+1) = cheby(l,x(i))
            else
                Ay(i+1,l+1) = ddcheby(l,x(i))
            end if
        end do
    end do

    ! ================================================= !
    !   Defining the x_source vector. 

    do i = 0 , N
        if ((i.eq.0).or.(i.eq.N)) then
            bx(i + 1) = 0.0d0
        else 
            bx(i + 1) =  
        end if
    end do

    ! ================================================= !
    !   Defining the x_source vector. 

    do i = 0 , N
        if ((i.eq.0).or.(i.eq.N)) then
            bx(i + 1) = 0.0d0
        else 
            bx(i + 1) = f(ox,oy)
        end if
    end do

end subroutine Vecs