#include <stdio.h>
#include <stdbool.h>

int main()
{

    /*
        ternary operator ? = shorthand for if-else statements

        (condition) ? value_if_true : vale_if_false;
    */

    int x = 5;
    int y = 6;

    bool isOnline = true;

    int max = (x > y)? x: y;
    int number = 8;

    printf("%d\n", max);
    printf("%s\n", (isOnline) ? "online" : "offline");
    printf("%d is %s\n", number, (number %2 == 0) ? "even":"odd");
    return 0;
}