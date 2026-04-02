#include <stdio.h>

void birthday(int* age);

int main()
{

    /*
        pointer = A variable that stores the memory address of another variable.
        Benefit: They yelp avoid wasting memory by allowing you to pass
        the address of a large data structure insted of copying the entire data.
    */

     int age = 25;
    //  int *pAge = &age;// * is the dereference operator

     printf("%p\n", &age); //prints where is it locate in memory

    //  birthday(pAge);
    birthday(&age);

     printf("You are %d years old ", age);

    return 0;
}

void birthday(int* age)
{
    // pas by reference
    (*age)++;
}