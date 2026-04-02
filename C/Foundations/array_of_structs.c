#include <stdio.h>

typedef struct
{
    char model[25];
    int year;
    int price;
}Car;

int main()
{

    Car cars[] = { {"Mustang", 2025, 32000},
                {"corvette", 2026, 68000},
                {"Challenger", 2024, 29000}};

    int number = sizeof(cars) / sizeof(cars[0]);

    for(int i =0; i < number; i++)
    {
        printf("%s %d $%d\n", cars[i].model, cars[i].year, cars[1].price);
    }


    return 0;
}