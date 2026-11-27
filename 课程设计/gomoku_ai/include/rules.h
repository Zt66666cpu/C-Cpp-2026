#ifndef GOMOKU_RULES_H
#define GOMOKU_RULES_H

#include "board.h"

typedef enum {
    MOVE_INVALID = 0,
    MOVE_OK,
    MOVE_BLACK_WIN,
    MOVE_WHITE_WIN,
    MOVE_BLACK_FORBIDDEN,
    MOVE_DRAW
} MoveResult;

typedef enum {
    FORBIDDEN_NONE = 0,
    FORBIDDEN_OVERLINE,
    FORBIDDEN_DOUBLE_FOUR,
    FORBIDDEN_DOUBLE_THREE
} ForbiddenReason;

typedef struct {
    MoveResult result;
    ForbiddenReason forbidden;
} MoveOutcome;

MoveOutcome rules_apply_move(
    Board *board,
    int row,
    int col,
    Cell player
);

/* 无副作用探针接口：试落子检查后保证恢复棋盘原状 */
bool rules_is_winning_move(Board *board, int row, int col, Cell player);
ForbiddenReason rules_check_black_forbidden(Board *board, int row, int col);
bool rules_is_legal_move(Board *board, int row, int col, Cell player);

const char *rules_forbidden_reason_name(ForbiddenReason reason);

#endif
