program data_gen
    implicit none
    integer :: i
    real :: x, y
    integer :: unit
    character(len=20) :: filename

    filename = 'data.txt'
    unit = 10
    
    open(unit=unit, file=filename, status='replace') ! Open the file

    do i = -10, 10
        x = real(i)
        y = x**2
        write(unit, '(f10.4, 2x, f10.4)') x, y ! Write x and y to the file
    end do
    
    close(unit) ! Close the file
    
    print *, "Data written to data.txt"
    print *, "Now run 'gnuplot -persist plot.plt' in your terminal"

end program data_gen
