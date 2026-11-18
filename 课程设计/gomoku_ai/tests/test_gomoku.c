#include "board.h"
#include "rules.h"

#include <stdio.h>

static int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            fprintf(                                                         \
                stderr,                                                      \
                "FAIL %s:%d: %s\n",                                         \
                __FILE__,                                                    \
                __LINE__,                                                    \
                #condition                                                   \
            );                                                               \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

static void test_board_basics(void) {
    Board board;
    board_init(&board);

    CHECK(board_stone_count(&board) == 0);
    CHECK(board_is_inside(0, 0));
    CHECK(board_is_inside(BOARD_SIZE - 1, BOARD_SIZE - 1));
    CHECK(!board_is_inside(-1, 0));
    CHECK(!board_is_inside(0, BOARD_SIZE));

    CHECK(board_place(&board, 7, 7, CELL_BLACK));
    CHECK(board.cells[7][7] == CELL_BLACK);
    CHECK(!board_place(&board, 7, 7, CELL_WHITE));
    CHECK(board_stone_count(&board) == 1);
}

static void test_black_horizontal_win(void) {
    Board board;
    board_init(&board);
    board.cells[7][3] = CELL_BLACK;
    board.cells[7][4] = CELL_BLACK;
    board.cells[7][5] = CELL_BLACK;
    board.cells[7][6] = CELL_BLACK;

    MoveOutcome outcome = rules_apply_move(&board, 7, 7, CELL_BLACK);

    CHECK(outcome.result == MOVE_BLACK_WIN);
    CHECK(outcome.forbidden == FORBIDDEN_NONE);
}

static void test_white_vertical_win(void) {
    Board board;
    board_init(&board);
    board.cells[3][7] = CELL_WHITE;
    board.cells[4][7] = CELL_WHITE;
    board.cells[5][7] = CELL_WHITE;
    board.cells[6][7] = CELL_WHITE;

    MoveOutcome outcome = rules_apply_move(&board, 7, 7, CELL_WHITE);

    CHECK(outcome.result == MOVE_WHITE_WIN);
}

static void test_diagonal_wins(void) {
    Board board;
    board_init(&board);
    board.cells[2][2] = CELL_BLACK;
    board.cells[3][3] = CELL_BLACK;
    board.cells[4][4] = CELL_BLACK;
    board.cells[5][5] = CELL_BLACK;
    CHECK(rules_apply_move(&board, 6, 6, CELL_BLACK).result
          == MOVE_BLACK_WIN);

    board_init(&board);
    board.cells[2][10] = CELL_WHITE;
    board.cells[3][9] = CELL_WHITE;
    board.cells[4][8] = CELL_WHITE;
    board.cells[5][7] = CELL_WHITE;
    CHECK(rules_apply_move(&board, 6, 6, CELL_WHITE).result
          == MOVE_WHITE_WIN);
}

static void test_black_overline_is_forbidden(void) {
    Board board;
    board_init(&board);
    board.cells[7][1] = CELL_BLACK;
    board.cells[7][2] = CELL_BLACK;
    board.cells[7][3] = CELL_BLACK;
    board.cells[7][4] = CELL_BLACK;
    board.cells[7][5] = CELL_BLACK;

    MoveOutcome outcome = rules_apply_move(&board, 7, 6, CELL_BLACK);

    CHECK(outcome.result == MOVE_BLACK_FORBIDDEN);
    CHECK(outcome.forbidden == FORBIDDEN_OVERLINE);
}

static void test_white_overline_wins(void) {
    Board board;
    board_init(&board);
    board.cells[7][1] = CELL_WHITE;
    board.cells[7][2] = CELL_WHITE;
    board.cells[7][3] = CELL_WHITE;
    board.cells[7][4] = CELL_WHITE;
    board.cells[7][5] = CELL_WHITE;

    MoveOutcome outcome = rules_apply_move(&board, 7, 6, CELL_WHITE);

    CHECK(outcome.result == MOVE_WHITE_WIN);
}

static void test_double_four_is_forbidden(void) {
    Board board;
    board_init(&board);
    board.cells[7][5] = CELL_BLACK;
    board.cells[7][6] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;
    board.cells[5][7] = CELL_BLACK;
    board.cells[6][7] = CELL_BLACK;
    board.cells[8][7] = CELL_BLACK;

    MoveOutcome outcome = rules_apply_move(&board, 7, 7, CELL_BLACK);

    CHECK(outcome.result == MOVE_BLACK_FORBIDDEN);
    CHECK(outcome.forbidden == FORBIDDEN_DOUBLE_FOUR);
}

static void test_single_four_is_allowed(void) {
    Board board;
    board_init(&board);
    board.cells[7][5] = CELL_BLACK;
    board.cells[7][6] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;

    MoveOutcome outcome = rules_apply_move(&board, 7, 7, CELL_BLACK);

    CHECK(outcome.result == MOVE_OK);
}

static void test_double_three_is_forbidden(void) {
    Board board;
    board_init(&board);
    board.cells[7][6] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;
    board.cells[6][7] = CELL_BLACK;
    board.cells[8][7] = CELL_BLACK;

    MoveOutcome outcome = rules_apply_move(&board, 7, 7, CELL_BLACK);

    CHECK(outcome.result == MOVE_BLACK_FORBIDDEN);
    CHECK(outcome.forbidden == FORBIDDEN_DOUBLE_THREE);
}

static void test_single_three_is_allowed(void) {
    Board board;
    board_init(&board);
    board.cells[7][6] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;

    MoveOutcome outcome = rules_apply_move(&board, 7, 7, CELL_BLACK);

    CHECK(outcome.result == MOVE_OK);
}

static void test_full_board_draw(void) {
    Board board;
    board_init(&board);

    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (row == BOARD_SIZE - 1 && col == BOARD_SIZE - 2) {
                continue;
            }

            int pattern = (2 * row + col) % 4;
            board.cells[row][col] =
                pattern < 2 ? CELL_BLACK : CELL_WHITE;
        }
    }

    MoveOutcome outcome = rules_apply_move(
        &board,
        BOARD_SIZE - 1,
        BOARD_SIZE - 2,
        CELL_BLACK
    );

    CHECK(outcome.result == MOVE_DRAW);
}

static void test_invalid_move_does_not_change_board(void) {
    Board board;
    board_init(&board);
    board.cells[7][7] = CELL_WHITE;

    MoveOutcome outcome = rules_apply_move(&board, 7, 7, CELL_BLACK);

    CHECK(outcome.result == MOVE_INVALID);
    CHECK(board.cells[7][7] == CELL_WHITE);
    CHECK(board_stone_count(&board) == 1);
}

int main(void) {
    test_board_basics();
    test_black_horizontal_win();
    test_white_vertical_win();
    test_diagonal_wins();
    test_black_overline_is_forbidden();
    test_white_overline_wins();
    test_double_four_is_forbidden();
    test_single_four_is_allowed();
    test_double_three_is_forbidden();
    test_single_three_is_allowed();
    test_full_board_draw();
    test_invalid_move_does_not_change_board();

    if (failures != 0) {
        fprintf(stderr, "%d test(s) failed.\n", failures);
        return 1;
    }

    printf("All Gomoku tests passed.\n");
    return 0;
}
