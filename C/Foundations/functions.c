#include <stdio.h>
#include <string.h>

void happyBirthday(char name[], int age)
{
    printf("\nHappy birthday to you!");
    printf("\nHappy birthday to you!");
    printf("\nHappy birthday dear %s!", name);
    printf("\nHappy birthday to you!");
    printf("\n you are %d years old\n", age);
}

int main()
{
    /*
        function = A reusable section of code that can be invoked "called"
        Arguments can be sent to a function so that it can use them
    */

    char name[50] = "Bro" ;
    int age = 25;

    printf("Enter your name\n");
    fgets(name, sizeof(name), stdin);
    name[strlen(name)-1]= '\0';

    printf("Enter your age: ");
    scanf("%d", &age);

    happyBirthday(name, age); //The order must be the same type and order


    return 0;
}