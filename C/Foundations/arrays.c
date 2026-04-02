#include <stdio.h>

int main()
{
    /*
    array = A fixed-size collection of elements of the same data type 
    similar to a variable, but it holds more than 1 value.
    */

    int numbers[] = {10, 20, 30, 40, 50};
    char grades[] = {'A', 'B', 'C', 'D', 'F'};

    printf("%lu\n", sizeof(numbers)); //Prints the size of the element in bytes
    printf("%d\n", numbers[2]);
    printf("%c\n", grades[4]);

    return 0;
}