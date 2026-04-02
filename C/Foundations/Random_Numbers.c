#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main()
{
    /*
        Pseudo-random = Appear random but are determined by a mathematical formula
        that uses a seed value to generate a predictable sequence of numbers. 
        Advanced: Mersenne Twister of /dev/random
    */

    srand(time(NULL)); //We're setting the seed for the random variable

    int min = 50;
    int max = 106;

    int randomNum = (rand() % (max - min + 1) )+ min; // Random number between min and max

    printf("%d\n", rand());
    printf("%d\n", RAND_MAX); //Prints the max random number depends on the compiler
    printf("%d\n", randomNum);

    return 0;
}