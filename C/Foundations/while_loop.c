#include <stdio.h>
#include <string.h>
#include <stdbool.h>

int main()
{
    /*
        while loop = continue some code WHILE the condition remains true
        Condition must be true for us to enter while loop
    */

    int number = 0;
    int number2 = 0;
    char name[50] = " ";
    bool isRunning = true;
    char response = '\0';

    printf("Enter your name ");
    fgets(name, sizeof(name),stdin);
    name[strlen(name)-1]= '\0';

    while(number <= 0)
    {
        printf("Enter a number greater than 0: ");
        scanf("%d", &number);
    }

    do
    {
        printf("Enter a number greater than 10: ");
        scanf("%d", &number2);
    } while (number2 <= 10);
    
    while(strlen(name) == 0)
    {
        printf("Name cannot be empty! please enter your name: ");
        fgets(name, sizeof(name),stdin);
        name[strlen(name)-1]= '\0';
    }
    printf("\nHello bro %s ",name);

    while(isRunning)
    {
        printf("You are playing a game\n");
        printf("Would you like to continue? Y = yes, N = no: ");
        scanf(" %c", &response);

        if(response != 'Y' && response != 'y')
        {
            isRunning = false;
        }
    }
    printf("You exited the game");

    return 0;
}