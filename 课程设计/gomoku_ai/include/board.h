#ifndef GOMOKU_BOARD_H
#define GOMOKU_BOARD_H

#include <stdbool.h>

#define BOARD_SIZE 15

typedef enum {
    CELL_EMPTY = 0,
    CELL_BLACK = 1,
    CELL_WHITE = 2
} Cell;

typedef struct {
    Cell cells[BOARD_SIZE][BOARD_SIZE];
} Board;

void board_init(Board *board);
bool board_is_inside(int row, int col);
bool board_is_empty(const Board *board, int row, int col);
bool board_place(Board *board, int row, int col, Cell player);
bool board_is_full(const Board *board);
int board_stone_count(const Board *board);

#endif
