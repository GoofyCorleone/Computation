#include <stdio.h>

int main()
{

    FILE *pFile = fopen("output.txt", "w"); // "r" is for reading

    char text[] = "MALDITA SEA PETRO Y LA MADRE";

    if(pFile == NULL)
    {
        printf("Erorr opening file\n");
        return 1;
    }

    fprintf(pFile, "%s", text); // file printf = fprintf 

    printf("File has been written succesfully!\;");

    fclose(pFile);

    return 0;
}