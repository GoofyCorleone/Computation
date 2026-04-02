#include <stdio.h>
#include <stdbool.h>


double square(double num)
{
    int result = num * num;
    return result;
}

bool ageCheck(int age)
{
    if(age >= 18 )
    {
        return true;
    }
    else
    {
        return false;
    }
}

int getMax(int x, int y)
{
    if(x >=y)
    {
        return x;
    } 
    else
    {
        return y;
    }
}

int main()
{

    // return = retuns a value back to where you call a function

    double x = square(2);
    double y = square(3.5);
    double z = square(4.4);
    int age = 21;
    int max = getMax(3,5);

    if (ageCheck(age))
    {
        printf("You may sign up\n");
    }
    else
    {
        printf("You must be 18+ to sign up\n");
    }

    printf("The max number is %d\n", max);
    printf("%lf\n",x);
    printf("%lf\n",y);
    printf("%lf\n",z);

    return 0;
}