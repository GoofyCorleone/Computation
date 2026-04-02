#include <stdio.h>
#include <stdbool.h>

void hello(char name[], int age); //Function prototype
bool ageCheck(int age);

int main()
{
    /*
        Function prototype = Prive the compiler W/ information about a function's:
        name, return type, adn parameters before its actual definition.
        Enables type checking and allows functiosn to be used before they're define.
        Improves readability, organization, and helps prevent errors.
    */

    hello("spongebob", 30);

    if(ageCheck(30))
    {
        printf("You are enough to work at the Krusty Krab");
    }
    else
    {
        printf("You must be 16+ to work ath the Krusty Krab");
    }

    return 0;
}

void hello(char name[], int age)
{
     printf("Hello %s\n", name);
     printf("Your are %d years old \n", age);
}

bool ageCheck(int age)
{
    if(age >=16)
    {
        return age >= 16;
    }
}