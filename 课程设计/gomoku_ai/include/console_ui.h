#ifndef GOMOKU_CONSOLE_UI_H
#define GOMOKU_CONSOLE_UI_H

#include "board.h"
#include "rules.h"
#include "search.h"

typedef enum {
    INPUT_OK,
    INPUT_INVALID,
    INPUT_QUIT,
    INPUT_EOF
} InputResult;

typedef enum {
    GAME_MODE_HUMAN_BLACK = 1,
    GAME_MODE_HUMAN_WHITE = 2,
    GAME_MODE_TWO_PLAYER = 3,
    GAME_MODE_AI_VS_AI = 4
} GameMode;

void ui_print_board(const Board *board);
void ui_print_board_with_last(const Board *board, Move last_move);
void ui_prompt_mode(void);
bool ui_read_mode(GameMode *mode);
void ui_prompt_move(Cell player);
InputResult ui_read_move(int *row, int *col);
void ui_print_ai_move(Cell player, Move move, const SearchStats *stats);
void ui_print_move_error(MoveOutcome outcome);
void ui_print_game_result(MoveOutcome outcome);

#endif
