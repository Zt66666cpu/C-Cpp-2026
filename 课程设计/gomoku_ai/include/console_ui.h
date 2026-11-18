#ifndef GOMOKU_CONSOLE_UI_H
#define GOMOKU_CONSOLE_UI_H

#include "board.h"
#include "rules.h"

typedef enum {
    INPUT_OK,
    INPUT_INVALID,
    INPUT_QUIT,
    INPUT_EOF
} InputResult;

void ui_print_board(const Board *board);
void ui_prompt_move(Cell player);
InputResult ui_read_move(int *row, int *col);
void ui_print_move_error(MoveOutcome outcome);
void ui_print_game_result(MoveOutcome outcome);

#endif
