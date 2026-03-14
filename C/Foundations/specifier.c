#include <stdio.h>

/*
    Format specifier = Special tokens that begin with a % symbol,
    followed by a character that specifies the dad and optional
    modifiers (width, precision, flags)
    they control how data is displayed or interpreted

*/

int main(){

    int age = 25;
    float price = 19.99;
    double pi = 3.1415926359;
    char currency = '$';
    char name[] = "Goofy Corleone";

    printf("%d\n",age);
    printf("%f\n",price);
    printf("%lf\n",pi);
    printf("%c \n",currency);
    printf("%s \n", name);

    // Width, the minus sign push then to left

    int num1 = 1;
    int num2 = 10;
    int num3 = 100;

    printf(" %3d\n ", num1);
    printf(" %04d\n ", num2);
    printf(" %-4d\n ", num3);

    // Precision

    float price1 = 19.99;
    float price2 = 1.50;
    float price3 = -100.00;

    printf("%.2f\n", price1);
    printf("%7.3f\n", price2); //combine width with precision
    printf("%+2.4f\n", price3);

    return 0;
}