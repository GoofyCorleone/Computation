#include <stdio.h>
#include <string.h>
#include <stdbool.h>

void grid(char content1[], char content2[], char content3[])
{
    char ladder[] = "---------";
    
    for(int i = 0; i < 5; i++)
    { 
        if (i == 0 || i == 4)
        {
            printf("%s\n",ladder);
        }
        else if( i == 1)
        {
            printf("%s\n",content1);

        }
        else if( i == 2)
        {
            printf("%s\n",content2);

        }
        else
        {
            printf("%s\n",content3);

        }
    }
    
}

bool WhoWins(char player1, char player2, char content1[], char content2[], char content3[])
{
    char aux1[] = "| @ @ @ |";
    char aux2[] = "| @ @ @ |";
    bool gameOn = true;

    for (int i = 0; i<3 ; i++)
    {
        for (int j = 2; j<8; j+=2)
        {
            aux1[j] = player1;
            aux2[j] = player2;
        }
    }

    if (strcmp(aux1, content1) == 0 || strcmp(aux1, content2) == 0 || strcmp(aux1, content3) == 0)
    {
        gameOn = false;
        printf("The player 1 wins\n");
        return gameOn;
    }
    else if (strcmp(aux2, content1) == 0 || strcmp(aux2, content2) == 0 || strcmp(aux2, content3) == 0)
    {
        gameOn = false;
        printf("The player 1 wins\n");
        return gameOn;
    }

    for (int i = 2; i<8 ; i+=2)
    {
        if( content1[i]==player1 && content2[i]==player1 && content3[i]==player1)
        {
            gameOn = false;
            printf("The player 1 wins\n");
            return gameOn;
        }
        else if( content1[i]==player2 && content2[i]==player2 && content3[i]==player2)
        {
            gameOn = false;
            printf("The player 2 wins\n");
            return gameOn;
        }
    }

    if (content1[2]== player1 && content2[4]== player1 && content3[6]== player1)
    {
        gameOn = false;
        printf("The player 1 wins\n");
        return gameOn;
    }
    else if (content1[2]== player2 && content2[4]== player2 && content3[6]== player2)
    {
        gameOn = false;
        printf("The player 2 wins\n");
        return gameOn;
    }
    else if (content1[6]== player1 && content2[4]== player1 && content3[2]== player1)
    {
        gameOn = false;
        printf("The player 1 wins\n");
        return gameOn;
    }
    else if (content1[6]== player2 && content2[4]== player2 && content3[2]== player2)
    {
        gameOn = false;
        printf("The player 2 wins\n");
        return gameOn;
    }
    return gameOn;
}

void GameStart(bool gameOn)
{
    bool Turn1 = true;
    bool Turn2 = false;

    char player1 = '\0';
    char player2 = '\0';

    char content1[] = "| @ @ @ |";
    char content2[] = "| @ @ @ |";
    char content3[] = "| @ @ @ |";

    int round = 0;
    int position1x = 0;
    int position2x = 0;
    int position1y = 0;
    int position2y = 0;

    while (gameOn)
    {
        if (round == 10)
        {
            printf("It was a draw, end of the game!\n");
            break;
        }
        else if (round == 0)
        {   

            printf("Round number 1!; Choose your fighter\n");

            printf("\nPlayer 1 choose your character: ");
            getchar();
            scanf("%c", &player1);
            printf("\nPlayer 1 you choose: %c\n", player1);

            printf("\nPlayer 2 choose your character: ");
            getchar();
            scanf("%c", &player2);
            printf("\nPlayer 2 you choose: %c\n", player2);

            round += 1;
        }
        else
        {
            if (Turn1)
            {
                printf("\nRound number %d!; Choose your position\n",round);
                printf("Player 1: Choose the position as an int like (ij), i,j=1,2,3\n");
                printf("Position in y: ");
                scanf("%d", &position1y);
                printf("\nPosition in x: ");
                scanf("%d", &position1x);

                if (position1x > 3 || position1y > 3 )
                {
                    printf("Select a correct position i = 1,2,3 or j=1,2,3\n");
                    continue;
                }

                for(int i = 0; i<3;i++)
                {
                    if (i == 0 && i == position1y -1 )
                    {
                        content1[position1x*2] = player1;
                    }
                    else if (i == 1 && i == position1y -1)
                    {
                        content2[position1x*2] = player1;
                    }
                    else if (i == 2 && i == position1y-1)
                    {
                        content3[position1x*2] = player1;
                    }
                }
                Turn1 = false;
                Turn2 = true;
                round +=1;

                grid(content1,content2,content3);
                gameOn = WhoWins(player1, player2, content1, content2, content3);
            }
            else
            {
                printf("\nRound number %d!; Choose your position\n",round);
                printf("Player 2: Choose the position as an int like (ij), i,j=1,2,3\n");
                printf("Position in y: ");
                scanf("%d", &position2y);
                printf("\nPosition in x: ");
                scanf("%d", &position2x);
                
                if (position1x == position2x && position1y == position2y)
                {
                    printf("Cannot be placed in the same position, try again!\n");
                    continue;
                }
                else if (position2x > 3 || position2y > 3 )
                {
                    printf("Select a correct position i = 1,2,3 or j=1,2,3\n");
                    continue;
                }

                for(int i = 0; i<3;i++)
                {
                    if (i == 0 && i == position2y-1)
                    {
                        content1[position2x*2] = player2;
                    }
                    else if (i == 1 && i == position2y-1)
                    {
                        content2[position2x*2] = player2;
                    }
                    else if (i == 2 && i == position2y-1)
                    {
                        content3[position2x*2] = player2;
                    }
                }
                Turn1 = true;
                Turn2 = false;
                round +=1;

                grid(content1,content2,content3);
                gameOn = WhoWins(player1, player2, content1, content2, content3);
            }
        }
        
    }
    
    
}

int main()
{
    /*
        Part 1: Creating the grid. The gird is going to have the following geometry

        ---------
        | @ @ @ |
        | @ @ @ |
        | @ @ @ |
        ---------
    */

    char StartGame = '\0';
    char player1 = '\0';
    char player2 = '\0';
    bool gameOn = true;

    printf("Welcom to the tiktaktoe game in C, do you wanna start the game: Y or N: ");
    scanf("%c", &StartGame);

    if(StartGame == 'Y' || StartGame == 'y')
    {
        grid("| @ @ @ |", "| @ @ @ |", "| @ @ @ |");
        GameStart(gameOn);
    }
    else
    {
        printf("End of the game, fuck off :3\n");
    }
    
    return 0;
} 