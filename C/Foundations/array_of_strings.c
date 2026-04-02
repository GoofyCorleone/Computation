#include <stdio.h>
#include <string.h>

int main()
{

    char fruits[][10] = {"Apple",
                         "Banana", 
                         "Coconut"};

    char names[3][25] = {0};

    int size = sizeof(fruits) / sizeof(fruits[0]);

    fruits[0][0] = 'e';
    fruits[0][4] = 'A';

    fruits[1][0] = 'a';
    fruits[1][5] = 'B';

    fruits[2][0] = 't';
    fruits[2][6] = 'C';

    for(int i = 0; i < size; i++)
    {
        printf("%s\n", fruits[i]);
    }
    
    printf("enter a name: ");
    fgets(names[0], sizeof(names[0]), stdin);
    names[0][strlen(names[0])-1] = '\0';

    printf("%s\n", names[0]);
    return 0;
}