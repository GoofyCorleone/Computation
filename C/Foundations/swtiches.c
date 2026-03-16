#include <stdio.h>

int main()
{
    /*
        Swtich = an alternative for using many if-else statements
        more efficient with fixedi nteger values
    */

    // int dayOfweek = 0;
    char dayOfweek = '\0';

    // printf("Enter a day of the week [1-7]: \n");
    // scanf("%d", &dayOfweek);
    printf("Enter a day of the week (M,T,W,R,F,S,U): ");
    scanf("%c", &dayOfweek);

    // switch(dayOfweek)
    // {
    //     case 1:
    //         printf("It is monday\n");
    //         break;
    //     case 2:
    //         printf("It is Tuesday\n");
    //         break;
    //     case 3:
    //         printf("It is wednesday\n");
    //         break;
    //     case 4:
    //         printf("It is Thursday\n");
    //         break;
    //     case 5:
    //         printf("It is fryday\n");
    //         break;
    //     case 6:
    //         printf("It is saturday\n");
    //         break;
    //     case 7:
    //         printf("It is Sunday\n");
    //         break;
    //     default:
    //         printf("Please only a number [1-7] \n");
    //         break;
    // }
    // return 0;

    switch(dayOfweek)
    {
        case 'M':
            printf("It is monday\n");
            break;
        case 'T':
            printf("It is Tuesday\n");
            break;
        case 'W':
            printf("It is wednesday\n");
            break;
        case 'R':
            printf("It is Thursday\n");
            break;
        case 'F':
            printf("It is fryday\n");
            break;
        case 'S':
            printf("It is saturday\n");
            break;
        case 'U':
            printf("It is Sunday\n");
            break;
        default:
            printf("Please only a number character (M,T,W,R,F,S,U) \n");
            break;
    }
    return 0;
}