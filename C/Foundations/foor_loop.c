#include <stdio.h>
#include <unistd.h>

int main()
{
    /* for loop = Repeat some code a limited $ of times
    for(Initizalization; condition; update)

    break = Break out of a loop (STOP)
    continue = skip current cycle of a loop (SKIP)
    */

    printf("First loop\n");
    for(int i= 0; i < 10; i++)
    {
        printf("%d\n", i);
    }

    printf("Second loop\n");
    for(int i= 1; i < 10; i+=2)
    {
        printf("%d\n", i);
    }

    printf("Third loop\n");
    for(int i = 10; i >= 0; i--)
    {
        sleep(1); //It is in seconds
        printf("%d\n", i);
    }
    printf("Happy New year\n");
    return 0;
}