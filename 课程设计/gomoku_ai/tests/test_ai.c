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
    cfg.max_depth = 3;
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
    cfg.max_depth = 3;
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
    cfg.max_depth = 3;
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

static void test_zobrist_consistency(void) {
    Board b1;
    Board b2;
    board_init(&b1);
    board_init(&b2);

    /* 相同局面不同落子顺序应得到相同 Zobrist 哈希 */
    board_place(&b1, 7, 7, CELL_BLACK);
    board_place(&b1, 7, 8, CELL_WHITE);
    board_place(&b1, 8, 7, CELL_BLACK);

    board_place(&b2, 8, 7, CELL_BLACK);
    board_place(&b2, 7, 8, CELL_WHITE);
    board_place(&b2, 7, 7, CELL_BLACK);

    uint64_t h1 = search_compute_zobrist(&b1, CELL_WHITE);
    uint64_t h2 = search_compute_zobrist(&b2, CELL_WHITE);
    uint64_t h3 = search_compute_zobrist(&b2, CELL_BLACK);
    CHECK(h1 == h2);
    CHECK(h1 != h3);
}

static void test_vcf_three_step_win(void) {
    /* 白方通过先冲四逼黑应子再成活四的三手 VCF 胜型 */
    Board board;
    board_init(&board);

    /* 横线第 7 行：白 (7,5)(7,6)(7,7)，左端被黑 (7,4) 挡，右端空 (7,8)(7,9) */
    board.cells[7][4] = CELL_BLACK;
    board.cells[7][5] = CELL_WHITE;
    board.cells[7][6] = CELL_WHITE;
    board.cells[7][7] = CELL_WHITE;
    /* 若白走 (7,8) 冲四，黑被迫堵 (7,9)；随后白在 (5,8)(6,8)(8,8) -> 竖向成活四 */
    board.cells[5][8] = CELL_WHITE;
    board.cells[6][8] = CELL_WHITE;
    board.cells[8][8] = CELL_WHITE;
    /* 给黑子放几颗无关紧要的子避免误判 */
    board.cells[3][3] = CELL_BLACK;
    board.cells[11][11] = CELL_BLACK;

    Move win_move = {-1, -1};
    uint64_t nodes = 0;
    bool found = search_find_vcf(&board, CELL_WHITE, 6, 1.0, &win_move, &nodes);
    CHECK(found);
    CHECK(win_move.row == 7 && win_move.col == 8);

    SearchConfig cfg = search_default_config();
    cfg.max_depth = 3;
    cfg.vcf_max_depth = 6;
    Move last = {11, 11};
    SearchResult res = search_find_best_move(&board, CELL_WHITE, last, &cfg, NULL, NULL);
    CHECK(res.valid);
    CHECK(res.stats.used_vcf);
    CHECK(res.best_move.row == 7 && res.best_move.col == 8);
}

static void test_block_open_three(void) {
    Board board;
    board_init(&board);
    board.cells[7][6] = CELL_WHITE;
    board.cells[7][7] = CELL_WHITE;
    board.cells[7][8] = CELL_WHITE;
    board.cells[10][10] = CELL_BLACK;
    board.cells[10][11] = CELL_BLACK;

    SearchConfig cfg = search_default_config();
    cfg.max_depth = 4;
    cfg.time_limit_sec = 2.0;

    Move last = {7, 8};
    SearchResult res = search_find_best_move(&board, CELL_BLACK, last, &cfg, NULL, NULL);
    CHECK(res.valid);
    CHECK(res.best_move.row == 7 && (res.best_move.col == 5 || res.best_move.col == 9));
}

int main(void) {
    test_board_undo_and_rule_probes();
    test_search_immediate_win();
    test_search_must_block_opponent_four();
    test_search_black_avoids_forbidden_move();
    test_eval_symmetry_and_features();
    test_zobrist_consistency();
    test_vcf_three_step_win();
    test_block_open_three();

    if (failures != 0) {
        fprintf(stderr, "%d AI test(s) failed.\n", failures);
        return 1;
    }

    printf("All Gomoku AI tests passed.\n");
    return 0;
}
