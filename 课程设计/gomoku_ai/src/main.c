#include <stdio.h>

#define BOARD_SIZE 15
#define BLACK 0
#define WHITE 1

void printBoard(int(* arrs)[15]);




int main(void) {


        int board[BOARD_SIZE][BOARD_SIZE];
        for (int i = 0; i < BOARD_SIZE; i++) {
            for (int j = 0; j < BOARD_SIZE; j++) {
                board[i][j] = 2;
            }
        }
    while (1) {
        printBoard(board);//显示棋盘
        return 0;
    }
}
