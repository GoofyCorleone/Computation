#include <stdio.h>

enum Status
{
    SUCCES, FAILURE, PENDING
};

void conectStatus(enum Status status);

int main()
{

    enum Status status = SUCCES;

    conectStatus(status);

    return 0;
}

void conectStatus(enum Status status)
{
    switch (status)
    {
    case SUCCES:
        printf("conecction was successfull\n");
        break;
    case FAILURE:
        printf("conecction was Failure\n");
        break;
    case PENDING:
        printf("Coneccting.\n");
        break;
    }
}