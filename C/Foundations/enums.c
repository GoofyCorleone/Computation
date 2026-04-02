#include <stdio.h>

// typedef enum
// {
//     SUNDAY = 1 , MONDAY = 2, TUESDAY = 3, WENESDAY = 4, THURSDAY = 5, FRIDAY = 6, SATURDAY = 7 //by default starts from 0
// }Day;

enum Day
{
    SUNDAY = 1 , MONDAY = 2, TUESDAY = 3, WENESDAY = 4, THURSDAY = 5, FRIDAY = 6, SATURDAY = 7 //by default starts from 0
};

int main()
{
    /*
        enum = A user-defined data type that consists
        of a set of named integer cosntants.
        Benefit: Replaces numbers with readable names

        SUNDAY = 0;
        MONDAY = 1;
        TUESDAY = 2;
    */

    enum Day today = SATURDAY;
    
    if( today == SUNDAY || today = SATURDAY)
    {
        printf("It's the weenkend");
    }
    else
    {
        printf("It is a weekday");
    }
    return 0;
}