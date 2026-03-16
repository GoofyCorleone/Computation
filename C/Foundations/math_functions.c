#include <stdio.h>
#include <math.h>

int main(){

    float x = 3.14;
    float y = 2.5;

    //x = sqrt(x);
    //x = pow(x,3); //Power
    //x = round(x);
    //x = ceil(x); 
    x = floor(x);

    //y = abs(y); //works only for integer it is not in math.h, apparently
    //y = log(y);
    //y = sin(x); //there is cos, tan

    printf("%f\n",x);
    printf("%f\n",y);

    return 0;
}