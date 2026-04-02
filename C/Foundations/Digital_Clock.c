#include <stdio.h>
#include <time.h>
#include <stdbool.h>
#include <unistd.h>

int main()
{

    time_t rawtime = 0; // Jan 1 1970 (EPoCH)
    struct tm *pTime = NULL;
    bool isRunning = true;

    printf("DIGITAL CLOCK\n");

    while (isRunning)
    {
        time(&rawtime);

        pTime = localtime(&rawtime); // Pointer to a struct

        printf("%02d:%02d:%02d\r", pTime->tm_hour, pTime->tm_min, pTime->tm_sec );
        fflush(stdout);
       // printf("%ld\n", rawtime); // ld = long decimal

        sleep(1);
    }
    

    return 0;
}