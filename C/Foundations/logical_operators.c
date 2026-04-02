#include <stdio.h>

int main()
{
    /*
    Logical operator = used to combine or modify boolean expression.

    && = AND
    || = OR
    ! = Not
    */

    int temp = 1000000000;

    if(temp > 0 && temp < 30)
    {
        printf("The temperature is good\n");
    }
    else if (temp < 30)
    {
        printf("The temperature is good\n");
    }
    else
    {
        printf("The temperetaure is bad");
    }

    return 0;
}