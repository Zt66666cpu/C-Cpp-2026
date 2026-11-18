#include "board.h"

void board_init(Board *board) {
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            board->cells[row][col] = CELL_EMPTY;
        }
    }
}

bool board_is_inside(int row, int col) {
    return row >= 0
        && row < BOARD_SIZE
        && col >= 0
        && col < BOARD_SIZE;
}

bool board_is_empty(const Board *board, int row, int col) {
    return board_is_inside(row, col)
        && board->cells[row][col] == CELL_EMPTY;
}

bool board_place(Board *board, int row, int col, Cell player) {
    if (player != CELL_BLACK && player != CELL_WHITE) {
        return false;
    }

    if (!board_is_empty(board, row, col)) {
        return false;
    }

    board->cells[row][col] = player;
    return true;
}

bool board_is_full(const Board *board) {
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board->cells[row][col] == CELL_EMPTY) {
                return false;
            }
        }
    }

    return true;
}

int board_stone_count(const Board *board) {
    int count = 0;

    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board->cells[row][col] != CELL_EMPTY) {
                ++count;
            }
        }
    }

    return count;
}
