#include <stdio.h>
#include <stdbool.h>
#include <string.h>

int main()
{
    int age = 0;

    printf("Enter your age: ");
    scanf("%d", &age);

    if(age >=65)
    {
        printf("You are a senior\n");
    }
    else if(age >=18)
    {
        printf("You are and adult\n");
    }
    else if(age < 0)
    {
        printf("You haven't born yet\n");
    }
    else if(age==0)
    {
        printf("You are a newborn\n");
    }
    else
    {
        printf("Your are a child\n");
    }

    bool isStudent = true;

    if(isStudent == true)
    {
        printf("You are a student\n");
    }
    else
    {
        printf("Your are not a student\n");
    }

    char name[50] = "";

    getchar();
    printf("Enter you name: ");
    fgets(name, sizeof(name),stdin);
    name[strlen(name)-1]= '\0';

    if(strlen(name)==0)
    {
        printf("You did not enter your name");
    }
    else
    {
        printf("Hello %s\n", name);
    }

    return 0;
}