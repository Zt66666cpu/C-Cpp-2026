#include "search.h"

#include "eval.h"
#include "rules.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CANDIDATES 64

typedef struct {
    Move move;
    int score;
} ScoredMove;

typedef struct {
    double deadline;
    bool time_up;
    uint64_t nodes;
    Cell root_player;
} SearchState;

static double now_seconds(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_scored_moves_desc(const void *a, const void *b) {
    const ScoredMove *ma = (const ScoredMove *)a;
    const ScoredMove *mb = (const ScoredMove *)b;
    if (ma->score < mb->score) return 1;
    if (ma->score > mb->score) return -1;
    return 0;
}

static bool has_neighbor_within(const Board *board, int row, int col, int radius) {
    int r_min = row - radius;
    int r_max = row + radius;
    int c_min = col - radius;
    int c_max = col + radius;

    if (r_min < 0) r_min = 0;
    if (r_max >= BOARD_SIZE) r_max = BOARD_SIZE - 1;
    if (c_min < 0) c_min = 0;
    if (c_max >= BOARD_SIZE) c_max = BOARD_SIZE - 1;

    for (int r = r_min; r <= r_max; ++r) {
        for (int c = c_min; c <= c_max; ++c) {
            if (board->cells[r][c] != CELL_EMPTY) {
                return true;
            }
        }
    }
    return false;
}

SearchConfig search_default_config(void) {
    SearchConfig cfg = {
        .time_limit_sec = 55.0,
        .max_depth = 4,
        .vcf_max_depth = 10,
        .use_nnue = true,
        .use_cnn = true,
        .cnn_value_weight = 0.35f,
        .temperature = 0.0f,
        .seed = 42ULL
    };
    return cfg;
}

int search_generate_candidates(
    Board *board,
    Cell player,
    int radius,
    Move *out_moves,
    int max_moves
) {
    if (max_moves <= 0) {
        return 0;
    }

    if (board_stone_count(board) == 0) {
        out_moves[0].row = BOARD_SIZE / 2;
        out_moves[0].col = BOARD_SIZE / 2;
        return 1;
    }

    ScoredMove scored[BOARD_CELLS];
    int count = 0;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (!board_is_empty(board, r, c)) {
                continue;
            }
            if (!has_neighbor_within(board, r, c, radius)) {
                continue;
            }
            if (!rules_is_legal_move(board, r, c, player)) {
                continue;
            }
            scored[count].move.row = r;
            scored[count].move.col = c;
            scored[count].score = eval_move_heuristic(board, r, c, player);
            ++count;
        }
    }

    /* 若局部邻域内全被占满或全是黑方禁手，则回退扫描全盘合法空位 */
    if (count == 0) {
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                if (rules_is_legal_move(board, r, c, player)) {
                    scored[count].move.row = r;
                    scored[count].move.col = c;
                    scored[count].score = eval_move_heuristic(board, r, c, player);
                    ++count;
                }
            }
        }
    }

    qsort(scored, (size_t)count, sizeof(ScoredMove), compare_scored_moves_desc);

    int out_count = count < max_moves ? count : max_moves;
    for (int i = 0; i < out_count; ++i) {
        out_moves[i] = scored[i].move;
    }
    return out_count;
}

/* Layer 1 战术护栏：检测一步必胜或对方成五点必防 */
static bool check_tactical_guardrail(
    Board *board,
    Cell player,
    Move *out_move,
    int *out_score
) {
    /* 1. 我方存在一步成五必胜点 */
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (rules_is_winning_move(board, r, c, player)) {
                out_move->row = r;
                out_move->col = c;
                *out_score = EVAL_MATE_SCORE - 1;
                return true;
            }
        }
    }

    /* 2. 对手存在立即成五威胁点（若只有一个或多个，必须优先拦截合法点） */
    Cell opp = board_opponent(player);
    Move defense_moves[BOARD_CELLS];
    int defense_count = 0;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (rules_is_winning_move(board, r, c, opp)
                && rules_is_legal_move(board, r, c, player)) {
                defense_moves[defense_count].row = r;
                defense_moves[defense_count].col = c;
                ++defense_count;
            }
        }
    }

    if (defense_count == 1) {
        *out_move = defense_moves[0];
        *out_score = eval_position(board, player);
        return true;
    }

    return false;
}

static int negamax(
    Board *board,
    Cell player,
    int depth,
    int ply,
    int alpha,
    int beta,
    SearchState *state
) {
    ++state->nodes;
    if ((state->nodes & 1023ULL) == 0ULL && now_seconds() >= state->deadline) {
        state->time_up = true;
        return 0;
    }

    if (board_is_full(board)) {
        return 0;
    }

    if (depth <= 0) {
        return eval_position(board, player);
    }

    Move candidates[20];
    int cand_count = search_generate_candidates(
        board, player, 2, candidates, depth >= 2 ? 18 : 14
    );
    if (cand_count == 0) {
        /* 当前行棋方无合法着法（全盘只有黑方禁手或无空位） */
        return -(EVAL_MATE_SCORE - ply);
    }

    int best = -EVAL_MATE_SCORE * 2;

    for (int i = 0; i < cand_count; ++i) {
        Move m = candidates[i];

        if (rules_is_winning_move(board, m.row, m.col, player)) {
            return EVAL_MATE_SCORE - ply - 1;
        }

        board_place(board, m.row, m.col, player);
        int score = -negamax(
            board,
            board_opponent(player),
            depth - 1,
            ply + 1,
            -beta,
            -alpha,
            state
        );
        board_undo(board, m.row, m.col);

        if (state->time_up) {
            return 0;
        }

        if (score > best) {
            best = score;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            break;
        }
    }

    return best;
}

SearchResult search_find_best_move(
    Board *board,
    Cell player,
    Move last_move,
    const SearchConfig *config,
    const NnueModel *nnue,
    const CnnModel *cnn
) {
    (void)last_move;
    (void)nnue;
    (void)cnn;

    SearchConfig cfg = config ? *config : search_default_config();
    double start_time = now_seconds();

    SearchResult result;
    memset(&result, 0, sizeof(result));
    result.best_move.row = -1;
    result.best_move.col = -1;

    if (player != CELL_BLACK && player != CELL_WHITE) {
        return result;
    }

    /* 空盘直接落天元 */
    if (board_stone_count(board) == 0) {
        result.best_move.row = BOARD_SIZE / 2;
        result.best_move.col = BOARD_SIZE / 2;
        result.root_policy[result.best_move.row * BOARD_SIZE + result.best_move.col] = 1.0f;
        result.valid = true;
        result.stats.used_guardrail = true;
        result.stats.completed_depth = 1;
        result.stats.nodes = 1;
        result.stats.elapsed_sec = now_seconds() - start_time;
        return result;
    }

    /* 保底合法候选点 */
    Move root_moves[24];
    int root_count = search_generate_candidates(board, player, 2, root_moves, 24);
    if (root_count == 0) {
        return result;
    }

    result.best_move = root_moves[0];
    result.valid = true;

    /* Layer 1 战术护栏 */
    Move guard_move;
    int guard_score = 0;
    if (check_tactical_guardrail(board, player, &guard_move, &guard_score)) {
        result.best_move = guard_move;
        result.root_policy[guard_move.row * BOARD_SIZE + guard_move.col] = 1.0f;
        result.stats.used_guardrail = true;
        result.stats.completed_depth = 1;
        result.stats.nodes = 1;
        result.stats.score = guard_score;
        result.stats.elapsed_sec = now_seconds() - start_time;
        return result;
    }

    int target_depth = cfg.max_depth < 1 ? 1 : cfg.max_depth;
    SearchState state = {
        .deadline = start_time + (cfg.time_limit_sec > 0.05 ? cfg.time_limit_sec : 0.05),
        .time_up = false,
        .nodes = 0,
        .root_player = player
    };

    int best_score = -EVAL_MATE_SCORE * 2;
    Move best_move = root_moves[0];
    int alpha = -EVAL_MATE_SCORE * 2;
    int beta = EVAL_MATE_SCORE * 2;

    for (int i = 0; i < root_count; ++i) {
        Move m = root_moves[i];
        board_place(board, m.row, m.col, player);
        int score = -negamax(
            board,
            board_opponent(player),
            target_depth - 1,
            1,
            -beta,
            -alpha,
            &state
        );
        board_undo(board, m.row, m.col);

        if (state.time_up) {
            break;
        }
        if (score > best_score) {
            best_score = score;
            best_move = m;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    result.best_move = best_move;
    result.root_policy[best_move.row * BOARD_SIZE + best_move.col] = 1.0f;
    result.stats.completed_depth = state.time_up ? 0 : target_depth;
    result.stats.nodes = state.nodes;
    result.stats.score = best_score;
    result.stats.elapsed_sec = now_seconds() - start_time;
    return result;
}
