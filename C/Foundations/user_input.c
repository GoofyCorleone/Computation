#include <stdio.h>
#include <string.h>

int main(){
    
    int age = 0;
    float gpa = 0.0f; // specifies a float zero
    char grade = '\0'; // null terminator
    char name[30] = ""; // 30 bytes or 30 characters

    printf("Enter your age\n");
    scanf("%d", &age);

    printf("Enter your gpa\n");
    scanf("%f", &gpa);

    printf("Enter your grade\n");
    scanf(" %c", &grade); //Adding the space tell C to ingnore the line gap

    getchar(); // Clears the last line gap
    printf("Enter your full name\n");
    //scanf("%s", &name); // Scanf cannot read blank spaces so we use fgets
    fgets(name, sizeof(name), stdin); // stdin = standart input
    name[strlen(name)-1] = '\0'; //makes the last caracet a null terminator because fgets puts an \n at the end

    printf("age= %d\n", age);
    printf("gpa= %f\n", gpa);
    printf("grade = %c\n", grade);
    printf("name= %s\n", name);
    return 0;
}