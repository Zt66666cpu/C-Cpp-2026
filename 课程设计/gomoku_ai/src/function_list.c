#include <stdio.h>

#define BOARD_SIZE 15
#define BLACK 0
#define WHITE 1

void printBoard(int board[BOARD_SIZE][BOARD_SIZE]) {
    int i, j;
    for (i = 0; i < BOARD_SIZE; i++) {
        for (j = 0; j < BOARD_SIZE; j++) {
            char symbol = '.';
            if (board[i][j] == BLACK) {
                symbol = 'X';
            } else if (board[i][j] == WHITE) {
                symbol = 'O';
            }
            printf("%c ", symbol);
        }
        printf("\n");
    }
    printf("\n");
}
