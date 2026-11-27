#include "board.h"
#include "eval.h"
#include "nn_model.h"
#include "pnnue.h"
#include "rules.h"
#include "search.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

static void test_pvs_root_verification_avoids_blunder(void) {
    /* 白方存在活三 (7,6)(7,7)(7,8)，黑方若不堵 (7,5) 或 (7,9) 将在 2 步内被成活四必败。
       启用 temperature > 0 验证根节点精确验证后，策略分布绝不把高概率分给漏防着法。 */
    Board board;
    board_init(&board);
    board.cells[7][6] = CELL_WHITE;
    board.cells[7][7] = CELL_WHITE;
    board.cells[7][8] = CELL_WHITE;
    board.cells[4][4] = CELL_BLACK;
    board.cells[4][5] = CELL_BLACK;
    board.cells[10][10] = CELL_BLACK;

    SearchConfig cfg = search_default_config();
    cfg.max_depth = 4;
    cfg.time_limit_sec = 2.0;
    cfg.temperature = 0.85f;
    cfg.seed = 99ULL;

    Move last = {7, 8};
    SearchResult res = search_find_best_move(&board, CELL_BLACK, last, &cfg, NULL, NULL);
    CHECK(res.valid);
    CHECK(res.best_move.row == 7 && (res.best_move.col == 5 || res.best_move.col == 9));

    float safe_mass = res.root_policy[7 * BOARD_SIZE + 5]
                    + res.root_policy[7 * BOARD_SIZE + 9];
    CHECK(safe_mass > 0.95f);
}

static void test_nn_missing_file_fallback(void) {
    CHECK(nnue_load("nonexistent_nnue_12345.bin") == NULL);
    CHECK(cnn_load("nonexistent_cnn_12345.bin") == NULL);

    Board board;
    board_init(&board);
    board.cells[7][7] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;
    board.cells[8][8] = CELL_WHITE;

    CHECK(eval_position_with_nnue(&board, CELL_BLACK, NULL)
          == eval_position(&board, CELL_BLACK));
}

static void test_vcf_black_avoids_forbidden_path(void) {
    /* 黑方若落 (7,7) 会在横竖两方向形成双四禁手，search_find_vcf 绝不可选中禁手点 */
    Board board;
    board_init(&board);
    board.cells[7][5] = CELL_BLACK;
    board.cells[7][6] = CELL_BLACK;
    board.cells[7][8] = CELL_BLACK;
    board.cells[5][7] = CELL_BLACK;
    board.cells[6][7] = CELL_BLACK;
    board.cells[8][7] = CELL_BLACK;
    board.cells[3][3] = CELL_WHITE;
    board.cells[3][4] = CELL_WHITE;

    Move win_move = {-1, -1};
    uint64_t nodes = 0;
    bool found = search_find_vcf(&board, CELL_BLACK, 6, 1.0, &win_move, &nodes);
    if (found) {
        CHECK(!(win_move.row == 7 && win_move.col == 7));
        CHECK(rules_is_legal_move(&board, win_move.row, win_move.col, CELL_BLACK));
    }
}

static void test_search_respects_tight_time_budget(void) {
    Board board;
    board_init(&board);
    board.cells[7][7] = CELL_BLACK;
    board.cells[7][8] = CELL_WHITE;
    board.cells[8][7] = CELL_BLACK;
    board.cells[6][6] = CELL_WHITE;
    board.cells[8][8] = CELL_BLACK;
    board.cells[6][8] = CELL_WHITE;
    board.cells[9][6] = CELL_BLACK;
    board.cells[9][7] = CELL_WHITE;

    SearchConfig cfg = search_default_config();
    cfg.max_depth = 6;
    cfg.time_limit_sec = 0.15;

    Move last = {9, 7};
    SearchResult res = search_find_best_move(&board, CELL_BLACK, last, &cfg, NULL, NULL);
    CHECK(res.valid);
    CHECK(rules_is_legal_move(&board, res.best_move.row, res.best_move.col, CELL_BLACK));
    CHECK(res.stats.elapsed_sec < 0.60);
}

static void test_eval_broken_four_and_jump_three(void) {
    Board board;
    board_init(&board);
    /* 第 7 行跳冲四：X X X . X */
    board.cells[7][3] = CELL_BLACK;
    board.cells[7][4] = CELL_BLACK;
    board.cells[7][5] = CELL_BLACK;
    board.cells[7][7] = CELL_BLACK;
    /* 第 9 行跳活三：. X . X X . */
    board.cells[9][4] = CELL_BLACK;
    board.cells[9][6] = CELL_BLACK;
    board.cells[9][7] = CELL_BLACK;

    EvalFeatures feat;
    eval_extract_features(&board, CELL_BLACK, &feat);
    CHECK(feat.values[4] >= 1.0f); /* my_rush_four */
    CHECK(feat.values[6] >= 1.0f); /* my_open_three */
}

static void test_jump_three_defense_and_cnn_smoke(void) {
    /* 跳活三 . O . O O . 应被及时识别并阻断 */
    Board board;
    board_init(&board);
    board.cells[7][5] = CELL_WHITE;
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
    CHECK(res.best_move.row == 7
          && (res.best_move.col == 6 || res.best_move.col == 4 || res.best_move.col == 9));

    /* 若运行目录下存在 models/cnn.bin，验证纯 C11 CNN 推理输出合法 */
    CnnModel *cnn = cnn_load("models/cnn.bin");
    if (cnn) {
        float cnn_in[BOARD_CELLS * CNN_INPUT_CHANNELS];
        float logits[BOARD_CELLS];
        nn_build_cnn_input(&board, CELL_BLACK, last, cnn_in);
        float val = cnn_forward(cnn, cnn_in, logits);
        CHECK(val >= -1.0f && val <= 1.0f);
        cnn_free(cnn);
    }
}

static void test_pnnue_incremental_equivalence(void) {
    pnnue_init_tables();

    PatternNnueModel *model = (PatternNnueModel *)calloc(1, sizeof(PatternNnueModel));
    CHECK(model != NULL);
    model->num_canon = PNNUE_NUM_CANON;
    model->dim = 16;
    model->v_hidden = 8;
    model->emb = (float *)calloc((size_t)model->num_canon * model->dim, sizeof(float));
    CHECK(model->emb != NULL);

    uint32_t rng = 0x20261127U;
    for (uint32_t p = 0; p < model->num_canon * model->dim; ++p) {
        rng = rng * 1664525U + 1013904223U;
        model->emb[p] = ((float)(rng & 0xFFFFU) / 65535.0f - 0.5f) * 0.25f;
    }
    for (uint32_t k = 0; k < model->dim; ++k) {
        model->cell_bias[k] = (k % 2 == 0) ? -0.02f : 0.03f;
        model->policy_w[0][k] = 0.05f * (float)(k + 1) / 16.0f;
        model->policy_w[1][k] = -0.04f * (float)(k + 1) / 16.0f;
    }
    model->policy_b[0] = 0.1f;
    model->policy_b[1] = -0.1f;
    for (uint32_t j = 0; j < model->v_hidden; ++j) {
        for (uint32_t k = 0; k < model->dim; ++k) {
            model->val_w1[j][k] = (((j + k) % 3) == 0 ? 0.08f : -0.05f);
        }
        model->val_b1[0][j] = 0.02f;
        model->val_b1[1][j] = -0.02f;
        model->val_w2[j] = (j % 2 == 0) ? 0.15f : -0.12f;
    }
    model->val_b2[0] = 0.05f;
    model->val_b2[1] = -0.05f;

    Board board;
    board_init(&board);
    PnnueAcc *inc = (PnnueAcc *)calloc(1, sizeof(PnnueAcc));
    PnnueAcc *ref = (PnnueAcc *)calloc(1, sizeof(PnnueAcc));
    CHECK(inc != NULL && ref != NULL);
    pnnue_acc_init(inc, model, &board);

    Move history[50];
    int n_moves = 0;
    for (int step = 0; step < 50; ++step) {
        Cell side = (step % 2 == 0) ? CELL_BLACK : CELL_WHITE;
        Move m = {-1, -1};
        for (int tries = 0; tries < 200; ++tries) {
            rng = rng * 1664525U + 1013904223U;
            int idx = (int)((rng >> 16) % BOARD_CELLS);
            int r = idx / BOARD_SIZE;
            int c = idx % BOARD_SIZE;
            if (board_is_empty(&board, r, c)) {
                m.row = r;
                m.col = c;
                break;
            }
        }
        CHECK(m.row >= 0);
        board_place(&board, m.row, m.col, side);
        pnnue_acc_update_cell(inc, model, m.row, m.col, side);
        history[n_moves++] = m;

        pnnue_acc_init(ref, model, &board);
        for (int i = 0; i < BOARD_CELLS; ++i) {
            for (int d = 0; d < PNNUE_DIRS; ++d) {
                CHECK(inc->pat_id[i][d] == ref->pat_id[i][d]);
            }
        }

        float v_inc_b = pnnue_eval_value(inc, model, CELL_BLACK);
        float v_ref_b = pnnue_eval_value(ref, model, CELL_BLACK);
        float v_inc_w = pnnue_eval_value(inc, model, CELL_WHITE);
        float v_ref_w = pnnue_eval_value(ref, model, CELL_WHITE);
        CHECK(fabsf(v_inc_b - v_ref_b) < 1e-5f);
        CHECK(fabsf(v_inc_w - v_ref_w) < 1e-5f);

        float p_inc[BOARD_CELLS];
        float p_ref[BOARD_CELLS];
        pnnue_eval_policy_all(inc, model, side, p_inc);
        pnnue_eval_policy_all(ref, model, side, p_ref);
        for (int i = 0; i < BOARD_CELLS; ++i) {
            CHECK(fabsf(p_inc[i] - p_ref[i]) < 1e-5f);
        }
    }

    for (int step = n_moves - 1; step >= 0; --step) {
        Move m = history[step];
        pnnue_acc_update_cell(inc, model, m.row, m.col, CELL_EMPTY);
        board_undo(&board, m.row, m.col);
    }
    pnnue_acc_init(ref, model, &board);
    CHECK(fabsf(pnnue_eval_value(inc, model, CELL_BLACK) - pnnue_eval_value(ref, model, CELL_BLACK)) < 1e-5f);

    /* 搜索入口必胜与必防校验 */
    board_init(&board);
    board.cells[7][3] = CELL_BLACK;
    board.cells[7][4] = CELL_BLACK;
    board.cells[7][5] = CELL_BLACK;
    board.cells[7][6] = CELL_BLACK;
    SearchConfig cfg = search_default_config();
    cfg.max_depth = 3;
    cfg.time_limit_sec = 1.0;
    SearchResult res = search_find_best_move_pnnue(&board, CELL_BLACK, (Move){7, 6}, &cfg, model);
    CHECK(res.valid);
    CHECK(res.best_move.row == 7 && (res.best_move.col == 2 || res.best_move.col == 7));

    free(inc);
    free(ref);
    pnnue_free(model);
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
    test_pvs_root_verification_avoids_blunder();
    test_nn_missing_file_fallback();
    test_vcf_black_avoids_forbidden_path();
    test_search_respects_tight_time_budget();
    test_eval_broken_four_and_jump_three();
    test_jump_three_defense_and_cnn_smoke();
    test_pnnue_incremental_equivalence();

    if (failures != 0) {
        fprintf(stderr, "%d AI test(s) failed.\n", failures);
        return 1;
    }

    printf("All Gomoku AI tests passed.\n");
    return 0;
}
