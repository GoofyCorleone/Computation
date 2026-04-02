#include <stdio.h>

typedef int Number;
typedef char string[50];

int main()
{
    // typeef existing_type new_name;

    // int x = 3;
    // int y = 4;
    // int z = x+y;

    Number x = 3;
    Number y = 4;
    Number z = x+y;

    printf("%d\n",z);

    string name = "Bro code";

    printf("%s\n", name);
    return 0;
}