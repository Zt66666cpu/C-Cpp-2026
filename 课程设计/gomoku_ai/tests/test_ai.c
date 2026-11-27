#include "board.h"
#include "eval.h"
#include "rules.h"
#include "search.h"

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

static void test_board_undo_and_rule_probes(void) {
    Board board;
    board_init(&board);

    CHECK(!board_undo(&board, 7, 7));
    CHECK(board_place(&board, 7, 7, CELL_BLACK));
    CHECK(board_undo(&board, 7, 7));
    CHECK(board_is_empty(&board, 7, 7));
    CHECK(board_stone_count(&board) == 0);

    /* 黑方双三禁手无副作用探针 */
    board.cells[7][6] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;
    board.cells[6][7] = CELL_BLACK;
    board.cells[8][7] = CELL_BLACK;

    CHECK(rules_check_black_forbidden(&board, 7, 7) == FORBIDDEN_DOUBLE_THREE);
    CHECK(!rules_is_legal_move(&board, 7, 7, CELL_BLACK));
    CHECK(rules_is_legal_move(&board, 7, 7, CELL_WHITE));
    CHECK(board_is_empty(&board, 7, 7));
}

static void test_search_immediate_win(void) {
    Board board;
    board_init(&board);
    board.cells[7][3] = CELL_BLACK;
    board.cells[7][4] = CELL_BLACK;
    board.cells[7][5] = CELL_BLACK;
    board.cells[7][6] = CELL_BLACK;
    board.cells[8][5] = CELL_WHITE;
    board.cells[8][6] = CELL_WHITE;
    board.cells[8][7] = CELL_WHITE;
    board.cells[8][8] = CELL_WHITE;

    SearchConfig cfg = search_default_config();
    cfg.max_depth = 2;
    cfg.time_limit_sec = 2.0;

    Move last = {8, 8};
    SearchResult res = search_find_best_move(&board, CELL_BLACK, last, &cfg, NULL, NULL);
    CHECK(res.valid);
    CHECK(res.best_move.row == 7 && (res.best_move.col == 2 || res.best_move.col == 7));
}

static void test_search_must_block_opponent_four(void) {
    Board board;
    board_init(&board);
    board.cells[7][3] = CELL_WHITE;
    board.cells[7][4] = CELL_WHITE;
    board.cells[7][5] = CELL_WHITE;
    board.cells[7][6] = CELL_WHITE;
    board.cells[7][7] = CELL_BLACK;

    SearchConfig cfg = search_default_config();
    cfg.max_depth = 2;
    cfg.time_limit_sec = 2.0;

    Move last = {7, 6};
    SearchResult res = search_find_best_move(&board, CELL_BLACK, last, &cfg, NULL, NULL);
    CHECK(res.valid);
    CHECK(res.best_move.row == 7 && res.best_move.col == 2);
}

static void test_search_black_avoids_forbidden_move(void) {
    Board board;
    board_init(&board);
    board.cells[7][6] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;
    board.cells[6][7] = CELL_BLACK;
    board.cells[8][7] = CELL_BLACK;
    board.cells[5][5] = CELL_WHITE;
    board.cells[5][6] = CELL_WHITE;
    board.cells[5][8] = CELL_WHITE;

    SearchConfig cfg = search_default_config();
    cfg.max_depth = 2;
    cfg.time_limit_sec = 2.0;

    Move last = {5, 8};
    SearchResult res = search_find_best_move(&board, CELL_BLACK, last, &cfg, NULL, NULL);
    CHECK(res.valid);
    CHECK(!(res.best_move.row == 7 && res.best_move.col == 7));
    CHECK(rules_is_legal_move(&board, res.best_move.row, res.best_move.col, CELL_BLACK));
}

static void test_eval_symmetry_and_features(void) {
    Board board;
    board_init(&board);
    board.cells[7][7] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;
    board.cells[7][9] = CELL_BLACK;
    board.cells[10][10] = CELL_WHITE;

    int black_score = eval_position(&board, CELL_BLACK);
    int white_score = eval_position(&board, CELL_WHITE);
    CHECK(black_score > 0);
    CHECK(white_score < 0);
}

int main(void) {
    test_board_undo_and_rule_probes();
    test_search_immediate_win();
    test_search_must_block_opponent_four();
    test_search_black_avoids_forbidden_move();
    test_eval_symmetry_and_features();

    if (failures != 0) {
        fprintf(stderr, "%d AI test(s) failed.\n", failures);
        return 1;
    }

    printf("All Gomoku AI tests passed.\n");
    return 0;
}
