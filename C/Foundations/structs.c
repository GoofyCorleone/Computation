#include <stdio.h>
#include <stdbool.h>

typedef struct 
{
    char name[50];
    int age;
    float gpa;
    bool isFullTime;
}Student ;


int main()
{
    /*
        struct = A custom container that holds ultiple pieces of related information.
        Similar to objects in other languages
    */

    Student student1 = {"Spongebob", 30, 2.5, true};

    printf("%s\n", student1.name);
    printf("%d\n", student1.age);
    printf("%.2ff\n", student1.gpa);
    printf("%s\n", (student1.isFullTime) ? "Yes" : "Nope");

    return 0;
}