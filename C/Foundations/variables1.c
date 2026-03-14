#include <stdio.h>  // for input and ouput
#include <stdbool.h> // for boolean

int main(){
    /*
     Variable = A resuble container for a value.
     Behaves as if it were the value it contains.

     %d is for integers
     %.mf is for float with m digits we can only have 67 digits after the dot
     %.mlf is for long float with m digits of precision
     %.c is for characters
     %s is for strings and strings are [] arrays of characters
     %d is for boolean

    */

    int age = 25;
    int year = 2025;
    int quantity = 1;

    printf("you are %d years old\n ", age);
    printf("The is year is %d\n",year);
    printf("You have ordered %d x items", quantity);

    float gpa = 2.5;
    float price = 19.99;
    float temperature = -10.1;

    printf("Your gpa is %.1f\n",gpa);
    printf("The prices is $%.3f\n", price);
    printf("The temperature is %.2f °C\n", temperature);

    double pi = 3.14159265359;
    double e = 2.718282828283;

    printf("The value of pi is %lf\n", pi);
    printf("Teh value of e is %.8lf\n", e);

    char grade = 'A' ;
    char symbol = '!';
    char currency = '$';

    printf("Your grade is %c\n",grade);
    printf("Your favorite symbol is %c\n", symbol);
    printf("The currency is %c\n",currency);

    char name[] = " Miguel Jafert ";
    char food[] = " Strawberries ";
    char email[] = "jafgofu1s@gmail.com";

    printf("My name is %s\n", name);
    printf("Your favorite food is %s\n", food);
    printf("My personal emain is %s\n",email);

    bool isOnline = true;

    if(isOnline){
        printf("Your are ONLINE\n");
    }
    else{
        printf("youre are OFFLINE\n");
    }

    printf(" The user is: %d\n ", isOnline);

    return 0;
}