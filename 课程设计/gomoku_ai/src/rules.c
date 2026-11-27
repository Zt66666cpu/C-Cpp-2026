#include "rules.h"

#include <stddef.h>

typedef struct {
    int row_step;
    int col_step;
} Direction;

static const Direction DIRECTIONS[] = {
    {0, 1},
    {1, 0},
    {1, 1},
    {1, -1}
};

static Cell cell_at(const Board *board, int row, int col) {
    if (!board_is_inside(row, col)) {
        return CELL_EMPTY;
    }

    return board->cells[row][col];
}

static void run_bounds(
    const Board *board,
    int base_row,
    int base_col,
    int base_step,
    Direction direction,
    Cell player,
    int *start_step,
    int *end_step
) {
    int start = base_step;
    int end = base_step;

    while (cell_at(
               board,
               base_row + direction.row_step * (start - 1),
               base_col + direction.col_step * (start - 1)
           ) == player) {
        --start;
    }

    while (cell_at(
               board,
               base_row + direction.row_step * (end + 1),
               base_col + direction.col_step * (end + 1)
           ) == player) {
        ++end;
    }

    *start_step = start;
    *end_step = end;
}

static int run_length_at(
    const Board *board,
    int row,
    int col,
    Direction direction,
    Cell player
) {
    int length = 1;

    for (int sign = -1; sign <= 1; sign += 2) {
        int step = 1;

        while (cell_at(
                   board,
                   row + direction.row_step * step * sign,
                   col + direction.col_step * step * sign
               ) == player) {
            ++length;
            ++step;
        }
    }

    return length;
}

static bool has_run_at_least(
    const Board *board,
    int row,
    int col,
    Cell player,
    int required_length
) {
    for (size_t i = 0; i < sizeof(DIRECTIONS) / sizeof(DIRECTIONS[0]); ++i) {
        if (run_length_at(board, row, col, DIRECTIONS[i], player)
            >= required_length) {
            return true;
        }
    }

    return false;
}

static bool has_exact_five(
    const Board *board,
    int row,
    int col,
    Cell player
) {
    for (size_t i = 0; i < sizeof(DIRECTIONS) / sizeof(DIRECTIONS[0]); ++i) {
        if (run_length_at(board, row, col, DIRECTIONS[i], player) == 5) {
            return true;
        }
    }

    return false;
}

static bool direction_has_four(
    Board *board,
    int row,
    int col,
    Direction direction,
    Cell player
) {
    for (int completion_step = -4; completion_step <= 4;
         ++completion_step) {
        int completion_row = row + direction.row_step * completion_step;
        int completion_col = col + direction.col_step * completion_step;

        if (!board_is_empty(board, completion_row, completion_col)) {
            continue;
        }

        board->cells[completion_row][completion_col] = player;

        int start_step;
        int end_step;
        run_bounds(
            board,
            row,
            col,
            completion_step,
            direction,
            player,
            &start_step,
            &end_step
        );

        bool includes_original_move = start_step <= 0 && end_step >= 0;
        bool creates_exact_five = end_step - start_step + 1 == 5;

        board->cells[completion_row][completion_col] = CELL_EMPTY;

        if (includes_original_move && creates_exact_five) {
            return true;
        }
    }

    return false;
}

static bool direction_has_open_three(
    Board *board,
    int row,
    int col,
    Direction direction,
    Cell player
) {
    for (int extension_step = -4; extension_step <= 4;
         ++extension_step) {
        int extension_row = row + direction.row_step * extension_step;
        int extension_col = col + direction.col_step * extension_step;

        if (!board_is_empty(board, extension_row, extension_col)) {
            continue;
        }

        board->cells[extension_row][extension_col] = player;
        int winning_points = 0;

        for (int completion_step = -4; completion_step <= 4;
             ++completion_step) {
            int completion_row = row + direction.row_step * completion_step;
            int completion_col = col + direction.col_step * completion_step;

            if (!board_is_empty(board, completion_row, completion_col)) {
                continue;
            }

            board->cells[completion_row][completion_col] = player;

            int start_step;
            int end_step;
            run_bounds(
                board,
                row,
                col,
                completion_step,
                direction,
                player,
                &start_step,
                &end_step
            );

            bool includes_original_move = start_step <= 0 && end_step >= 0;
            bool includes_extension =
                start_step <= extension_step && end_step >= extension_step;
            bool creates_exact_five = end_step - start_step + 1 == 5;

            board->cells[completion_row][completion_col] = CELL_EMPTY;

            if (includes_original_move
                && includes_extension
                && creates_exact_five) {
                ++winning_points;
            }
        }

        board->cells[extension_row][extension_col] = CELL_EMPTY;

        if (winning_points >= 2) {
            return true;
        }
    }

    return false;
}

static ForbiddenReason detect_black_forbidden(
    Board *board,
    int row,
    int col
) {
    if (has_run_at_least(board, row, col, CELL_BLACK, 6)) {
        return FORBIDDEN_OVERLINE;
    }

    int four_count = 0;
    int open_three_count = 0;

    for (size_t i = 0; i < sizeof(DIRECTIONS) / sizeof(DIRECTIONS[0]); ++i) {
        if (direction_has_four(
                board,
                row,
                col,
                DIRECTIONS[i],
                CELL_BLACK
            )) {
            ++four_count;
        } else if (direction_has_open_three(
                       board,
                       row,
                       col,
                       DIRECTIONS[i],
                       CELL_BLACK
                   )) {
            ++open_three_count;
        }
    }

    if (four_count >= 2) {
        return FORBIDDEN_DOUBLE_FOUR;
    }

    if (open_three_count >= 2) {
        return FORBIDDEN_DOUBLE_THREE;
    }

    return FORBIDDEN_NONE;
}

MoveOutcome rules_apply_move(
    Board *board,
    int row,
    int col,
    Cell player
) {
    MoveOutcome outcome = {
        .result = MOVE_INVALID,
        .forbidden = FORBIDDEN_NONE
    };

    if (!board_place(board, row, col, player)) {
        return outcome;
    }

    if (player == CELL_WHITE
        && has_run_at_least(board, row, col, CELL_WHITE, 5)) {
        outcome.result = MOVE_WHITE_WIN;
        return outcome;
    }

    if (player == CELL_BLACK) {
        if (has_exact_five(board, row, col, CELL_BLACK)) {
            outcome.result = MOVE_BLACK_WIN;
            return outcome;
        }

        outcome.forbidden = detect_black_forbidden(board, row, col);
        if (outcome.forbidden != FORBIDDEN_NONE) {
            outcome.result = MOVE_BLACK_FORBIDDEN;
            return outcome;
        }
    }

    if (board_is_full(board)) {
        outcome.result = MOVE_DRAW;
        return outcome;
    }

    outcome.result = MOVE_OK;
    return outcome;
}

bool rules_is_winning_move(Board *board, int row, int col, Cell player) {
    if ((player != CELL_BLACK && player != CELL_WHITE)
        || !board_is_empty(board, row, col)) {
        return false;
    }

    board->cells[row][col] = player;
    bool wins = (player == CELL_BLACK)
        ? has_exact_five(board, row, col, CELL_BLACK)
        : has_run_at_least(board, row, col, CELL_WHITE, 5);
    board->cells[row][col] = CELL_EMPTY;

    return wins;
}

ForbiddenReason rules_check_black_forbidden(Board *board, int row, int col) {
    if (!board_is_empty(board, row, col)) {
        return FORBIDDEN_NONE;
    }

    board->cells[row][col] = CELL_BLACK;
    ForbiddenReason reason = FORBIDDEN_NONE;
    if (!has_exact_five(board, row, col, CELL_BLACK)) {
        reason = detect_black_forbidden(board, row, col);
    }
    board->cells[row][col] = CELL_EMPTY;

    return reason;
}

bool rules_is_legal_move(Board *board, int row, int col, Cell player) {
    if ((player != CELL_BLACK && player != CELL_WHITE)
        || !board_is_empty(board, row, col)) {
        return false;
    }

    if (player == CELL_BLACK) {
        return rules_check_black_forbidden(board, row, col) == FORBIDDEN_NONE;
    }

    return true;
}

const char *rules_forbidden_reason_name(ForbiddenReason reason) {
    switch (reason) {
        case FORBIDDEN_OVERLINE:
            return "long line (six or more stones)";
        case FORBIDDEN_DOUBLE_FOUR:
            return "double four";
        case FORBIDDEN_DOUBLE_THREE:
            return "double open three";
        case FORBIDDEN_NONE:
        default:
            return "none";
    }
}
