#include "selfplay_api.h"

#include "board.h"
#include "eval.h"
#include "nn_model.h"
#include "pnnue.h"
#include "rules.h"
#include "search.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    NnueModel *nnue;
    CnnModel *cnn;
} ModelBundle;

static uint64_t next_rng(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void record_board_snapshot(
    Board *board,
    Cell player,
    Move last_move,
    uint8_t *dst_board,
    uint8_t *dst_forbidden,
    float *dst_features
) {
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int idx = r * BOARD_SIZE + c;
            dst_board[idx] = (uint8_t)board->cells[r][c];
            if (player == CELL_BLACK && board->cells[r][c] == CELL_EMPTY) {
                dst_forbidden[idx] =
                    (rules_check_black_forbidden(board, r, c) != FORBIDDEN_NONE)
                    ? 1U : 0U;
            } else {
                dst_forbidden[idx] = 0U;
            }
        }
    }

    EvalFeatures feat;
    eval_extract_features(board, player, &feat);
    memcpy(dst_features, feat.values, sizeof(float) * EVAL_FEATURE_DIM);
    (void)last_move;
}

static bool apply_random_opening_move(
    Board *board,
    Cell player,
    uint64_t *rng,
    Move *out_move
) {
    if (board_stone_count(board) == 0) {
        /* 首手在天元附近 5x5 内随机，增强开局覆盖 */
        int dr = (int)(next_rng(rng) % 5ULL) - 2;
        int dc = (int)(next_rng(rng) % 5ULL) - 2;
        int r = 7 + dr;
        int c = 7 + dc;
        if (rules_apply_move(board, r, c, player).result == MOVE_OK) {
            out_move->row = r;
            out_move->col = c;
            return true;
        }
    }

    Move cands[32];
    int n = search_generate_candidates(board, player, 2, cands, 16);
    if (n <= 0) {
        return false;
    }
    int pick = (int)(next_rng(rng) % (uint64_t)n);
    Move m = cands[pick];
    MoveOutcome out = rules_apply_move(board, m.row, m.col, player);
    if (out.result != MOVE_OK) {
        return false;
    }
    *out_move = m;
    return true;
}

void *gomoku_load_models(const char *nnue_path, const char *cnn_path) {
    ModelBundle *bundle = calloc(1, sizeof(ModelBundle));
    if (!bundle) return NULL;
    if (nnue_path && nnue_path[0] != '\0') {
        bundle->nnue = nnue_load(nnue_path);
    }
    if (cnn_path && cnn_path[0] != '\0') {
        bundle->cnn = cnn_load(cnn_path);
    }
    return bundle;
}

void gomoku_free_models(void *handle) {
    if (!handle) return;
    ModelBundle *bundle = (ModelBundle *)handle;
    nnue_free(bundle->nnue);
    cnn_free(bundle->cnn);
    free(bundle);
}

int gomoku_generate_game(
    void *models_handle,
    int max_depth,
    int vcf_max_depth,
    double time_limit_sec,
    int random_opening_plies,
    float explore_temp,
    int explore_plies,
    float cnn_value_weight,
    uint64_t seed,
    int max_steps,
    uint8_t *out_boards,
    uint8_t *out_players,
    int32_t *out_last_moves,
    uint8_t *out_forbidden,
    float *out_features,
    float *out_policies,
    float *out_search_values,
    float *out_outcomes
) {
    if (max_steps <= 0 || !out_boards || !out_players || !out_last_moves
        || !out_forbidden || !out_features || !out_policies
        || !out_search_values || !out_outcomes) {
        return -1;
    }

    ModelBundle *bundle = (ModelBundle *)models_handle;
    const NnueModel *nnue = bundle ? bundle->nnue : NULL;
    const CnnModel *cnn = bundle ? bundle->cnn : NULL;

    Board board;
    board_init(&board);
    uint64_t rng = seed ? seed : 0x123456789ABCDEF0ULL;

    Cell player = CELL_BLACK;
    Move last_move = {-1, -1};

    for (int p = 0; p < random_opening_plies; ++p) {
        Move om;
        if (!apply_random_opening_move(&board, player, &rng, &om)) {
            break;
        }
        last_move = om;
        player = board_opponent(player);
    }

    int steps = 0;
    Cell winner = CELL_EMPTY;

    while (steps < max_steps) {
        SearchConfig cfg = search_default_config();
        cfg.max_depth = max_depth > 0 ? max_depth : 3;
        cfg.vcf_max_depth = vcf_max_depth;
        cfg.time_limit_sec = time_limit_sec > 0.05 ? time_limit_sec : 0.5;
        cfg.use_nnue = (nnue != NULL);
        cfg.use_cnn = (cnn != NULL);
        cfg.cnn_value_weight = cnn_value_weight;
        cfg.temperature = (steps < explore_plies) ? explore_temp : 0.05f;
        cfg.seed = next_rng(&rng);

        record_board_snapshot(
            &board,
            player,
            last_move,
            out_boards + (size_t)steps * BOARD_CELLS,
            out_forbidden + (size_t)steps * BOARD_CELLS,
            out_features + (size_t)steps * EVAL_FEATURE_DIM
        );
        out_players[steps] = (uint8_t)player;
        out_last_moves[steps] = (last_move.row >= 0 && last_move.col >= 0)
            ? (last_move.row * BOARD_SIZE + last_move.col)
            : -1;

        SearchResult res = search_find_best_move(
            &board, player, last_move, &cfg, nnue, cnn
        );
        if (!res.valid) {
            winner = board_opponent(player);
            break;
        }

        memcpy(
            out_policies + (size_t)steps * BOARD_CELLS,
            res.root_policy,
            sizeof(float) * BOARD_CELLS
        );
        if (res.stats.score >= EVAL_MATE_SCORE - 1000) {
            out_search_values[steps] = 1.0f;
        } else if (res.stats.score <= -(EVAL_MATE_SCORE - 1000)) {
            out_search_values[steps] = -1.0f;
        } else {
            out_search_values[steps] = tanhf((float)res.stats.score / 25000.0f);
        }

        MoveOutcome outcome = rules_apply_move(
            &board, res.best_move.row, res.best_move.col, player
        );
        ++steps;

        if (outcome.result == MOVE_INVALID) {
            winner = board_opponent(player);
            break;
        }
        if (outcome.result == MOVE_BLACK_WIN) {
            winner = CELL_BLACK;
            break;
        }
        if (outcome.result == MOVE_WHITE_WIN || outcome.result == MOVE_BLACK_FORBIDDEN) {
            winner = CELL_WHITE;
            break;
        }
        if (outcome.result == MOVE_DRAW) {
            winner = CELL_EMPTY;
            break;
        }

        last_move = res.best_move;
        player = board_opponent(player);
    }

    for (int t = 0; t < steps; ++t) {
        if (winner == CELL_EMPTY) {
            out_outcomes[t] = 0.0f;
        } else if ((Cell)out_players[t] == winner) {
            out_outcomes[t] = 1.0f;
        } else {
            out_outcomes[t] = -1.0f;
        }
    }

    return steps;
}

int gomoku_run_match(
    void *models_a,
    int a_max_depth,
    int a_vcf_depth,
    double a_time_limit,
    float a_cnn_weight,
    void *models_b,
    int b_max_depth,
    int b_vcf_depth,
    double b_time_limit,
    float b_cnn_weight,
    int num_games,
    uint64_t seed,
    MatchStats *out_stats
) {
    if (!out_stats || num_games <= 0) {
        return -1;
    }
    memset(out_stats, 0, sizeof(*out_stats));

    ModelBundle *ba = (ModelBundle *)models_a;
    ModelBundle *bb = (ModelBundle *)models_b;

    double a_total_t = 0.0;
    double b_total_t = 0.0;
    int a_moves = 0;
    int b_moves = 0;

    uint64_t rng = seed ? seed : 0xCAFEBABE12345678ULL;
    /* 每对开局共享的对称随机第二手候选，避免 40 局重复同一盘 */
    static const int OPEN_OFFSETS[16][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
        {0, 1},   {1, -1}, {1, 0},  {1, 1},
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
        {1, -2},  {1, 2},  {2, -1}, {2, 1}
    };

    for (int g = 0; g < num_games; ++g) {
        int pair_idx = g / 2;
        bool a_is_black = (g % 2 == 0);

        Board board;
        board_init(&board);
        /* 成对对称开局：先落天元 (7,7)，白第 2 手按 pair_idx 选点 */
        rules_apply_move(&board, 7, 7, CELL_BLACK);
        int off_idx = (pair_idx + (int)(seed % 16ULL)) % 16;
        Move m2 = {7 + OPEN_OFFSETS[off_idx][0], 7 + OPEN_OFFSETS[off_idx][1]};
        rules_apply_move(&board, m2.row, m2.col, CELL_WHITE);

        Cell player = CELL_BLACK;
        Move last_move = m2;
        Cell winner = CELL_EMPTY;

        for (int ply = 2; ply < BOARD_CELLS; ++ply) {
            bool turn_a = (a_is_black && player == CELL_BLACK)
                || (!a_is_black && player == CELL_WHITE);

            const NnueModel *nnue = turn_a
                ? (ba ? ba->nnue : NULL)
                : (bb ? bb->nnue : NULL);
            const CnnModel *cnn = turn_a
                ? (ba ? ba->cnn : NULL)
                : (bb ? bb->cnn : NULL);

            SearchConfig cfg = search_default_config();
            cfg.max_depth = turn_a ? a_max_depth : b_max_depth;
            cfg.vcf_max_depth = turn_a ? a_vcf_depth : b_vcf_depth;
            cfg.time_limit_sec = turn_a ? a_time_limit : b_time_limit;
            cfg.cnn_value_weight = turn_a ? a_cnn_weight : b_cnn_weight;
            cfg.use_nnue = (nnue != NULL);
            cfg.use_cnn = (cnn != NULL);
            cfg.temperature = 0.0f;
            cfg.seed = next_rng(&rng);

            SearchResult res = search_find_best_move(
                &board, player, last_move, &cfg, nnue, cnn
            );

            if (turn_a) {
                a_total_t += res.stats.elapsed_sec;
                ++a_moves;
            } else {
                b_total_t += res.stats.elapsed_sec;
                ++b_moves;
            }

            if (!res.valid) {
                ++out_stats->illegal_moves;
                winner = board_opponent(player);
                break;
            }

            MoveOutcome outcome = rules_apply_move(
                &board, res.best_move.row, res.best_move.col, player
            );
            if (outcome.result == MOVE_INVALID) {
                ++out_stats->illegal_moves;
                winner = board_opponent(player);
                break;
            }
            if (outcome.result == MOVE_BLACK_WIN) {
                winner = CELL_BLACK;
                break;
            }
            if (outcome.result == MOVE_WHITE_WIN) {
                winner = CELL_WHITE;
                break;
            }
            if (outcome.result == MOVE_BLACK_FORBIDDEN) {
                ++out_stats->illegal_moves;
                winner = CELL_WHITE;
                break;
            }
            if (outcome.result == MOVE_DRAW) {
                winner = CELL_EMPTY;
                break;
            }

            last_move = res.best_move;
            player = board_opponent(player);
        }

        if (winner == CELL_EMPTY) {
            ++out_stats->draws;
        } else if ((a_is_black && winner == CELL_BLACK)
                   || (!a_is_black && winner == CELL_WHITE)) {
            ++out_stats->a_wins;
        } else {
            ++out_stats->b_wins;
        }
    }

    out_stats->a_avg_time_sec = a_moves > 0 ? a_total_t / a_moves : 0.0;
    out_stats->b_avg_time_sec = b_moves > 0 ? b_total_t / b_moves : 0.0;
    return 0;
}

int gomoku_eval_nn_debug(
    const uint8_t *board_cells,
    int player,
    int last_move_idx,
    const char *nnue_path,
    const char *cnn_path,
    float *out_cnn_input_hwc,
    float *out_features_32,
    float *out_nnue_val,
    float *out_cnn_val,
    float *out_cnn_logits
) {
    if (!board_cells) return -1;

    Board board;
    board_init(&board);
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            board.cells[r][c] = (Cell)board_cells[r * BOARD_SIZE + c];
        }
    }

    Cell side = (player == CELL_WHITE) ? CELL_WHITE : CELL_BLACK;
    Move last_move = {-1, -1};
    if (last_move_idx >= 0 && last_move_idx < BOARD_CELLS) {
        last_move.row = last_move_idx / BOARD_SIZE;
        last_move.col = last_move_idx % BOARD_SIZE;
    }

    EvalFeatures feat;
    eval_extract_features(&board, side, &feat);
    if (out_features_32) {
        memcpy(out_features_32, feat.values, sizeof(float) * EVAL_FEATURE_DIM);
    }

    float cnn_in[BOARD_CELLS * CNN_INPUT_CHANNELS];
    nn_build_cnn_input(&board, side, last_move, cnn_in);
    if (out_cnn_input_hwc) {
        memcpy(out_cnn_input_hwc, cnn_in, sizeof(cnn_in));
    }

    if (nnue_path && nnue_path[0] != '\0' && out_nnue_val) {
        NnueModel *nnue = nnue_load(nnue_path);
        if (!nnue) return -2;
        *out_nnue_val = nnue_forward(nnue, &feat);
        nnue_free(nnue);
    }

    if (cnn_path && cnn_path[0] != '\0' && out_cnn_val) {
        CnnModel *cnn = cnn_load(cnn_path);
        if (!cnn) return -3;
        *out_cnn_val = cnn_forward(cnn, cnn_in, out_cnn_logits);
        cnn_free(cnn);
    }

    return 0;
}

int gomoku_extract_state(
    const uint8_t *board_cells,
    int player,
    uint8_t *out_forbidden,
    float *out_features
) {
    if (!board_cells) return -1;

    Board board;
    board_init(&board);
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            board.cells[r][c] = (Cell)board_cells[r * BOARD_SIZE + c];
        }
    }

    Cell side = (player == CELL_WHITE) ? CELL_WHITE : CELL_BLACK;
    if (out_forbidden) {
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                int idx = r * BOARD_SIZE + c;
                if (side == CELL_BLACK && board.cells[r][c] == CELL_EMPTY) {
                    out_forbidden[idx] =
                        (rules_check_black_forbidden(&board, r, c) != FORBIDDEN_NONE)
                        ? 1U : 0U;
                } else {
                    out_forbidden[idx] = 0U;
                }
            }
        }
    }

    if (out_features) {
        EvalFeatures feat;
        eval_extract_features(&board, side, &feat);
        memcpy(out_features, feat.values, sizeof(float) * EVAL_FEATURE_DIM);
    }
    return 0;
}

int gomoku_step_search(
    void *models_handle,
    const uint8_t *board_cells,
    int player,
    int last_move_idx,
    int max_depth,
    int vcf_max_depth,
    double time_limit_sec,
    float explore_temp,
    float cnn_value_weight,
    uint64_t seed,
    int *out_best_move,
    float *out_search_value,
    float *out_root_policy,
    uint8_t *out_forbidden,
    float *out_features
) {
    if (!board_cells) return -1;

    Board board;
    board_init(&board);
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            board.cells[r][c] = (Cell)board_cells[r * BOARD_SIZE + c];
        }
    }

    Cell side = (player == CELL_WHITE) ? CELL_WHITE : CELL_BLACK;
    Move last_move = {-1, -1};
    if (last_move_idx >= 0 && last_move_idx < BOARD_CELLS) {
        last_move.row = last_move_idx / BOARD_SIZE;
        last_move.col = last_move_idx % BOARD_SIZE;
    }

    gomoku_extract_state(board_cells, (int)side, out_forbidden, out_features);

    ModelBundle *bundle = (ModelBundle *)models_handle;
    const NnueModel *nnue = bundle ? bundle->nnue : NULL;
    const CnnModel *cnn = bundle ? bundle->cnn : NULL;

    SearchConfig cfg = search_default_config();
    cfg.max_depth = max_depth > 0 ? max_depth : 3;
    cfg.vcf_max_depth = vcf_max_depth;
    cfg.time_limit_sec = time_limit_sec > 0.05 ? time_limit_sec : 0.5;
    cfg.use_nnue = (nnue != NULL);
    cfg.use_cnn = (cnn != NULL);
    cfg.cnn_value_weight = cnn_value_weight;
    cfg.temperature = explore_temp;
    cfg.seed = seed ? seed : 0x123456789ABCDEF0ULL;

    SearchResult res = search_find_best_move(
        &board, side, last_move, &cfg, nnue, cnn
    );
    if (!res.valid) {
        if (out_best_move) *out_best_move = -1;
        return -2;
    }

    if (out_best_move) {
        *out_best_move = res.best_move.row * BOARD_SIZE + res.best_move.col;
    }
    if (out_root_policy) {
        memcpy(out_root_policy, res.root_policy, sizeof(float) * BOARD_CELLS);
    }
    if (out_search_value) {
        if (res.stats.score >= EVAL_MATE_SCORE - 1000) {
            *out_search_value = 1.0f;
        } else if (res.stats.score <= -(EVAL_MATE_SCORE - 1000)) {
            *out_search_value = -1.0f;
        } else {
            *out_search_value = tanhf((float)res.stats.score / 25000.0f);
        }
    }
    return 0;
}

/* ==================== V3 Pattern-NNUE 导出接口 ==================== */

void *gomoku_load_pnnue(const char *pnnue_path) {
    if (!pnnue_path || pnnue_path[0] == '\0') return NULL;
    return (void *)pnnue_load(pnnue_path);
}

void gomoku_free_pnnue(void *handle) {
    pnnue_free((PatternNnueModel *)handle);
}

int gomoku_eval_pnnue_debug(
    const uint8_t *board_cells,
    int player,
    const char *pnnue_path,
    float *out_value,
    float *out_policy_logits
) {
    if (!board_cells || !pnnue_path) return -1;
    PatternNnueModel *model = pnnue_load(pnnue_path);
    if (!model) return -2;

    Board board;
    board_init(&board);
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            board.cells[r][c] = (Cell)board_cells[r * BOARD_SIZE + c];
        }
    }

    PnnueAcc *acc = (PnnueAcc *)calloc(1, sizeof(PnnueAcc));
    if (!acc) {
        pnnue_free(model);
        return -3;
    }

    Cell side = (player == CELL_WHITE) ? CELL_WHITE : CELL_BLACK;
    pnnue_acc_init(acc, model, &board);
    if (out_value) {
        *out_value = pnnue_eval_value(acc, model, side);
    }
    if (out_policy_logits) {
        pnnue_eval_policy_all(acc, model, side, out_policy_logits);
    }

    free(acc);
    pnnue_free(model);
    return 0;
}

int gomoku_generate_game_pnnue(
    void *pnnue_handle,
    int max_depth,
    int vcf_max_depth,
    double time_limit_sec,
    int random_opening_plies,
    float explore_temp,
    uint64_t seed,
    uint8_t *out_boards,
    uint8_t *out_players,
    uint8_t *out_forbidden,
    float *out_policies,
    float *out_values,
    float *out_outcomes
) {
    const PatternNnueModel *pnnue = (const PatternNnueModel *)pnnue_handle;
    Board board;
    board_init(&board);

    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    Cell player = CELL_BLACK;
    Move last_move = {-1, -1};
    Cell winner = CELL_EMPTY;
    int steps = 0;

    if (random_opening_plies < 0) random_opening_plies = 0;
    if (random_opening_plies > 12) random_opening_plies = 12;

    for (int p = 0; p < random_opening_plies; ++p) {
        Move om;
        if (!apply_random_opening_move(&board, player, &rng, &om)) {
            break;
        }
        last_move = om;
        player = board_opponent(player);
    }

    for (int ply = 0; ply < BOARD_CELLS; ++ply) {
        SearchConfig cfg = search_default_config();
        cfg.max_depth = max_depth > 0 ? max_depth : 3;
        cfg.vcf_max_depth = vcf_max_depth;
        cfg.time_limit_sec = time_limit_sec > 0.05 ? time_limit_sec : 0.5;
        cfg.use_nnue = (pnnue != NULL);
        cfg.use_cnn = false;
        cfg.temperature = (steps < 16) ? explore_temp : (explore_temp * 0.25f);
        cfg.seed = next_rng(&rng);

        SearchResult res = pnnue
            ? search_find_best_move_pnnue(&board, player, last_move, &cfg, pnnue)
            : search_find_best_move(&board, player, last_move, &cfg, NULL, NULL);
        if (!res.valid) {
            winner = board_opponent(player);
            break;
        }

        uint8_t *dst_board = out_boards + steps * BOARD_CELLS;
        uint8_t *dst_forb = out_forbidden + steps * BOARD_CELLS;
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                int idx = r * BOARD_SIZE + c;
                dst_board[idx] = (uint8_t)board.cells[r][c];
                if (player == CELL_BLACK && board.cells[r][c] == CELL_EMPTY) {
                    dst_forb[idx] =
                        (rules_check_black_forbidden(&board, r, c) != FORBIDDEN_NONE)
                        ? 1U : 0U;
                } else {
                    dst_forb[idx] = 0U;
                }
            }
        }
        out_players[steps] = (uint8_t)player;
        memcpy(out_policies + steps * BOARD_CELLS, res.root_policy, sizeof(float) * BOARD_CELLS);

        float val_norm = 0.0f;
        if (res.stats.score >= EVAL_MATE_SCORE - 1000) {
            val_norm = 1.0f;
        } else if (res.stats.score <= -(EVAL_MATE_SCORE - 1000)) {
            val_norm = -1.0f;
        } else {
            val_norm = tanhf((float)res.stats.score / 25000.0f);
        }
        out_values[steps] = val_norm;
        ++steps;

        MoveOutcome outcome = rules_apply_move(
            &board, res.best_move.row, res.best_move.col, player
        );
        if (outcome.result == MOVE_INVALID) {
            winner = board_opponent(player);
            break;
        }
        if (outcome.result == MOVE_BLACK_WIN) {
            winner = CELL_BLACK;
            break;
        }
        if (outcome.result == MOVE_WHITE_WIN || outcome.result == MOVE_BLACK_FORBIDDEN) {
            winner = CELL_WHITE;
            break;
        }
        if (outcome.result == MOVE_DRAW) {
            winner = CELL_EMPTY;
            break;
        }

        last_move = res.best_move;
        player = board_opponent(player);
    }

    for (int t = 0; t < steps; ++t) {
        if (winner == CELL_EMPTY) {
            out_outcomes[t] = 0.0f;
        } else if ((Cell)out_players[t] == winner) {
            out_outcomes[t] = 1.0f;
        } else {
            out_outcomes[t] = -1.0f;
        }
    }
    return steps;
}

int gomoku_run_match_pnnue(
    void *pnnue_a,
    int a_max_depth,
    int a_vcf_depth,
    double a_time_limit,
    void *pnnue_b,
    void *v2_models_b,
    int b_max_depth,
    int b_vcf_depth,
    double b_time_limit,
    float b_cnn_weight,
    int num_games,
    uint64_t seed,
    MatchStats *out_stats
) {
    if (!out_stats || num_games <= 0) return -1;
    memset(out_stats, 0, sizeof(*out_stats));

    const PatternNnueModel *pa = (const PatternNnueModel *)pnnue_a;
    const PatternNnueModel *pb = (const PatternNnueModel *)pnnue_b;
    ModelBundle *vb = (ModelBundle *)v2_models_b;

    double a_total_t = 0.0;
    double b_total_t = 0.0;
    int a_moves = 0;
    int b_moves = 0;
    uint64_t rng = seed ? seed : 0xCAFEBABE12345678ULL;

    static const int OPEN_OFFSETS[16][2] = {
        {-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
        {0, 1},   {1, -1}, {1, 0},  {1, 1},
        {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2},
        {1, -2},  {1, 2},  {2, -1}, {2, 1}
    };
    static const int OPEN_M3[4][2] = {
        {0, 2}, {2, 0}, {-2, 0}, {0, -2}
    };

    for (int g = 0; g < num_games; ++g) {
        int pair_idx = g / 2;
        bool a_is_black = (g % 2 == 0);

        Board board;
        board_init(&board);
        rules_apply_move(&board, 7, 7, CELL_BLACK);
        int off2 = (pair_idx + (int)(seed % 16ULL)) % 16;
        Move m2 = {7 + OPEN_OFFSETS[off2][0], 7 + OPEN_OFFSETS[off2][1]};
        rules_apply_move(&board, m2.row, m2.col, CELL_WHITE);

        Move last_move = m2;
        Cell player = CELL_BLACK;
        int start_ply = 2;

        int m3_base = ((pair_idx / 16) + (int)((seed >> 4) % 4ULL)) % 4;
        for (int t = 0; t < 4; ++t) {
            int idx3 = (m3_base + t) % 4;
            Move m3 = {7 + OPEN_M3[idx3][0], 7 + OPEN_M3[idx3][1]};
            if (m3.row == m2.row && m3.col == m2.col) continue;
            if (rules_is_legal_move(&board, m3.row, m3.col, CELL_BLACK)) {
                rules_apply_move(&board, m3.row, m3.col, CELL_BLACK);
                last_move = m3;
                player = CELL_WHITE;
                start_ply = 3;
                break;
            }
        }

        Cell winner = CELL_EMPTY;
        for (int ply = start_ply; ply < BOARD_CELLS; ++ply) {
            bool turn_a = (a_is_black && player == CELL_BLACK)
                || (!a_is_black && player == CELL_WHITE);

            SearchConfig cfg = search_default_config();
            cfg.temperature = 0.0f;
            cfg.seed = next_rng(&rng);

            SearchResult res;
            if (turn_a) {
                cfg.max_depth = a_max_depth;
                cfg.vcf_max_depth = a_vcf_depth;
                cfg.time_limit_sec = a_time_limit;
                cfg.use_nnue = (pa != NULL);
                cfg.use_cnn = false;
                res = pa
                    ? search_find_best_move_pnnue(&board, player, last_move, &cfg, pa)
                    : search_find_best_move(&board, player, last_move, &cfg, NULL, NULL);
                a_total_t += res.stats.elapsed_sec;
                ++a_moves;
            } else {
                cfg.max_depth = b_max_depth;
                cfg.vcf_max_depth = b_vcf_depth;
                cfg.time_limit_sec = b_time_limit;
                if (pb) {
                    cfg.use_nnue = true;
                    cfg.use_cnn = false;
                    res = search_find_best_move_pnnue(&board, player, last_move, &cfg, pb);
                } else {
                    const NnueModel *bn = vb ? vb->nnue : NULL;
                    const CnnModel *bc = vb ? vb->cnn : NULL;
                    cfg.use_nnue = (bn != NULL);
                    cfg.use_cnn = (bc != NULL);
                    cfg.cnn_value_weight = b_cnn_weight;
                    res = search_find_best_move(&board, player, last_move, &cfg, bn, bc);
                }
                b_total_t += res.stats.elapsed_sec;
                ++b_moves;
            }

            if (!res.valid) {
                ++out_stats->illegal_moves;
                winner = board_opponent(player);
                break;
            }

            MoveOutcome outcome = rules_apply_move(
                &board, res.best_move.row, res.best_move.col, player
            );
            if (outcome.result == MOVE_INVALID) {
                ++out_stats->illegal_moves;
                winner = board_opponent(player);
                break;
            }
            if (outcome.result == MOVE_BLACK_WIN) {
                winner = CELL_BLACK;
                break;
            }
            if (outcome.result == MOVE_WHITE_WIN) {
                winner = CELL_WHITE;
                break;
            }
            if (outcome.result == MOVE_BLACK_FORBIDDEN) {
                ++out_stats->illegal_moves;
                winner = CELL_WHITE;
                break;
            }
            if (outcome.result == MOVE_DRAW) {
                winner = CELL_EMPTY;
                break;
            }

            last_move = res.best_move;
            player = board_opponent(player);
        }

        if (winner == CELL_EMPTY) {
            ++out_stats->draws;
        } else if ((a_is_black && winner == CELL_BLACK)
                   || (!a_is_black && winner == CELL_WHITE)) {
            ++out_stats->a_wins;
        } else {
            ++out_stats->b_wins;
        }
    }

    out_stats->a_avg_time_sec = a_moves > 0 ? a_total_t / a_moves : 0.0;
    out_stats->b_avg_time_sec = b_moves > 0 ? b_total_t / b_moves : 0.0;
    return 0;
}
