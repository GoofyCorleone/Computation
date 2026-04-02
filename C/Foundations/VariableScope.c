#include <stdio.h>

int add(int x, int y)
{
    int result = x + y;
    return result;
}

int subtract(int x, int y)
{
    int result = x - y;
    return result;
}

int main()
{
    /*
        Variable scop = Refers to where a variable is recognized and accessisble.
        Variables can share the same name if they're in differente scopte {}
    */

    int result = add(3,4);
    int result2 = subtract(3,4);

    printf("The sum is %d\n", result);
    printf("The subtract is %d\n", result2);
    return 0;
}