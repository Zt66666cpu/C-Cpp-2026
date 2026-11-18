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

const char *rules_forbidden_reason_name(ForbiddenReason reason);

#endif
