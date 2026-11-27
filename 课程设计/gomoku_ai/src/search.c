#include "search.h"

#include "eval.h"
#include "nn_model.h"
#include "pnnue.h"
#include "rules.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

#define MAX_PLY 64
#define TT_SIZE (1U << 17)
#define TT_MASK (TT_SIZE - 1U)
#define MATE_BOUND (EVAL_MATE_SCORE - 1000)

typedef enum {
    TT_EMPTY = 0,
    TT_EXACT = 1,
    TT_LOWER = 2,
    TT_UPPER = 3
} TTFlag;

typedef struct {
    uint64_t key;
    int32_t score;
    int16_t depth;
    uint8_t flag;
    uint8_t best_move_idx;
} TTEntry;

typedef struct {
    Move move;
    int score;
} ScoredMove;

typedef struct {
    double deadline;
    bool time_up;
    uint64_t nodes;
    Cell root_player;
    uint64_t zobrist;
    TTEntry *tt;
    const NnueModel *nnue;
    Move killers[MAX_PLY][2];
    int history[2][BOARD_CELLS];
} SearchState;

static uint64_t g_zobrist_stone[BOARD_SIZE][BOARD_SIZE][3];
static uint64_t g_zobrist_side;
static bool g_zobrist_ready = false;

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void init_zobrist(void) {
    if (g_zobrist_ready) {
        return;
    }
    uint64_t seed = 0x20261127A1B2C3D4ULL;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            g_zobrist_stone[r][c][0] = 0ULL;
            g_zobrist_stone[r][c][1] = splitmix64(&seed);
            g_zobrist_stone[r][c][2] = splitmix64(&seed);
        }
    }
    g_zobrist_side = splitmix64(&seed);
    g_zobrist_ready = true;
}

uint64_t search_compute_zobrist(const Board *board, Cell side_to_move) {
    init_zobrist();
    uint64_t h = 0ULL;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            Cell cell = board->cells[r][c];
            if (cell == CELL_BLACK || cell == CELL_WHITE) {
                h ^= g_zobrist_stone[r][c][cell];
            }
        }
    }
    if (side_to_move == CELL_WHITE) {
        h ^= g_zobrist_side;
    }
    return h;
}

static double now_seconds(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
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

static int score_to_tt(int score, int ply) {
    if (score >= MATE_BOUND) return score + ply;
    if (score <= -MATE_BOUND) return score - ply;
    return score;
}

static int score_from_tt(int score, int ply) {
    if (score >= MATE_BOUND) return score - ply;
    if (score <= -MATE_BOUND) return score + ply;
    return score;
}

static bool same_move(Move a, Move b) {
    return a.row == b.row && a.col == b.col;
}

SearchConfig search_default_config(void) {
    SearchConfig cfg = {
        .time_limit_sec = 55.0,
        .max_depth = 5,
        .vcf_max_depth = 12,
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
            if (!board_is_empty(board, r, c)) continue;
            if (!has_neighbor_within(board, r, c, radius)) continue;
            if (!rules_is_legal_move(board, r, c, player)) continue;
            scored[count].move.row = r;
            scored[count].move.col = c;
            scored[count].score = eval_move_heuristic(board, r, c, player);
            ++count;
        }
    }

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

/* ==================== VCF 连续冲四杀棋求解器 ==================== */

static int collect_four_winning_points(
    Board *board,
    int row,
    int col,
    Cell attacker,
    Move *out_wins,
    int max_wins
) {
    static const int DR[4] = {0, 1, 1, 1};
    static const int DC[4] = {1, 0, 1, -1};

    int count = 0;
    for (int d = 0; d < 4; ++d) {
        for (int s = -4; s <= 4; ++s) {
            if (s == 0) continue;
            int nr = row + s * DR[d];
            int nc = col + s * DC[d];
            if (!board_is_empty(board, nr, nc)) continue;
            if (rules_is_winning_move(board, nr, nc, attacker)) {
                bool dup = false;
                for (int k = 0; k < count; ++k) {
                    if (out_wins[k].row == nr && out_wins[k].col == nc) {
                        dup = true;
                        break;
                    }
                }
                if (!dup && count < max_wins) {
                    out_wins[count].row = nr;
                    out_wins[count].col = nc;
                    ++count;
                }
            }
        }
    }
    return count;
}

static bool vcf_dfs(
    Board *board,
    Cell attacker,
    int depth_left,
    double deadline,
    Move *win_move,
    uint64_t *nodes,
    bool *time_up
) {
    ++(*nodes);
    if ((*nodes & 511ULL) == 0ULL && now_seconds() >= deadline) {
        *time_up = true;
        return false;
    }

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (rules_is_winning_move(board, r, c, attacker)) {
                if (win_move) {
                    win_move->row = r;
                    win_move->col = c;
                }
                return true;
            }
        }
    }

    if (depth_left <= 0 || *time_up) {
        return false;
    }

    Move cands[32];
    int cand_count = search_generate_candidates(board, attacker, 2, cands, 32);
    Cell defender = board_opponent(attacker);

    for (int i = 0; i < cand_count; ++i) {
        if (*time_up) return false;
        Move m = cands[i];
        board_place(board, m.row, m.col, attacker);

        Move win_pts[8];
        int wp_count = collect_four_winning_points(
            board, m.row, m.col, attacker, win_pts, 8
        );

        if (wp_count >= 2) {
            board_undo(board, m.row, m.col);
            if (win_move) *win_move = m;
            return true;
        }

        if (wp_count == 1) {
            Move def = win_pts[0];
            bool def_wins = rules_is_winning_move(board, def.row, def.col, defender);
            if (!def_wins) {
                if (!rules_is_legal_move(board, def.row, def.col, defender)) {
                    board_undo(board, m.row, m.col);
                    if (win_move) *win_move = m;
                    return true;
                }

                board_place(board, def.row, def.col, defender);
                Move dummy;
                bool sub_win = vcf_dfs(
                    board, attacker, depth_left - 2, deadline,
                    &dummy, nodes, time_up
                );
                board_undo(board, def.row, def.col);

                if (sub_win) {
                    board_undo(board, m.row, m.col);
                    if (win_move) *win_move = m;
                    return true;
                }
            }
        }

        board_undo(board, m.row, m.col);
    }

    return false;
}

bool search_find_vcf(
    Board *board,
    Cell attacker,
    int max_depth,
    double time_limit_sec,
    Move *win_move,
    uint64_t *out_nodes
) {
    if (attacker != CELL_BLACK && attacker != CELL_WHITE) {
        return false;
    }
    double deadline = now_seconds() + (time_limit_sec > 0.01 ? time_limit_sec : 0.01);
    uint64_t nodes = 0;
    bool time_up = false;
    bool found = vcf_dfs(
        board, attacker, max_depth, deadline, win_move, &nodes, &time_up
    );
    if (out_nodes) {
        *out_nodes = nodes;
    }
    return found;
}

/* ==================== 战术护栏与启发式排序 ==================== */

static bool check_tactical_guardrail(
    Board *board,
    Cell player,
    const NnueModel *nnue,
    Move *out_move,
    int *out_score
) {
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
        *out_score = eval_position_with_nnue(board, player, nnue);
        return true;
    }

    return false;
}

static int generate_ordered_moves(
    Board *board,
    Cell player,
    int ply,
    Move tt_move,
    SearchState *state,
    Move *out_moves,
    int max_moves
) {
    ScoredMove scored[BOARD_CELLS];
    int count = 0;
    int side_idx = (player == CELL_BLACK) ? 0 : 1;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (!board_is_empty(board, r, c)) continue;
            if (!has_neighbor_within(board, r, c, 2)) continue;
            if (!rules_is_legal_move(board, r, c, player)) continue;

            Move m = {r, c};
            int s = eval_move_heuristic(board, r, c, player);

            if (same_move(m, tt_move)) {
                s += 5000000;
            } else if (ply < MAX_PLY && same_move(m, state->killers[ply][0])) {
                s += 250000;
            } else if (ply < MAX_PLY && same_move(m, state->killers[ply][1])) {
                s += 180000;
            }
            s += state->history[side_idx][r * BOARD_SIZE + c];

            scored[count].move = m;
            scored[count].score = s;
            ++count;
        }
    }

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
    for (int i = out_count; i < count && out_count < max_moves + 4; ++i) {
        if (scored[i].score >= 2400) {
            out_count = i + 1;
        } else {
            break;
        }
    }

    for (int i = 0; i < out_count; ++i) {
        out_moves[i] = scored[i].move;
    }
    return out_count;
}

/* ==================== PVS + 置换表 Negamax 搜索 ==================== */

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

    int orig_alpha = alpha;
    TTEntry *tte = &state->tt[state->zobrist & TT_MASK];
    Move tt_move = {-1, -1};
    if (tte->flag != TT_EMPTY && tte->key == state->zobrist) {
        int idx = tte->best_move_idx;
        if (idx < BOARD_CELLS) {
            tt_move.row = idx / BOARD_SIZE;
            tt_move.col = idx % BOARD_SIZE;
        }
        if (tte->depth >= depth) {
            int tt_score = score_from_tt(tte->score, ply);
            if (tte->flag == TT_EXACT) return tt_score;
            if (tte->flag == TT_LOWER && tt_score > alpha) alpha = tt_score;
            else if (tte->flag == TT_UPPER && tt_score < beta) beta = tt_score;
            if (alpha >= beta) return tt_score;
        }
    }

    if (depth <= 0) {
        return eval_position_with_nnue(board, player, state->nnue);
    }

    Move candidates[24];
    int limit = depth >= 3 ? 18 : 14;
    int cand_count = generate_ordered_moves(
        board, player, ply, tt_move, state, candidates, limit
    );
    if (cand_count == 0) {
        return -(EVAL_MATE_SCORE - ply);
    }

    int best = -EVAL_MATE_SCORE * 2;
    Move best_move = candidates[0];
    int side_idx = (player == CELL_BLACK) ? 0 : 1;
    Cell opp = board_opponent(player);

    for (int i = 0; i < cand_count; ++i) {
        Move m = candidates[i];

        if (rules_is_winning_move(board, m.row, m.col, player)) {
            int win_score = EVAL_MATE_SCORE - ply - 1;
            tte->key = state->zobrist;
            tte->depth = (int16_t)depth;
            tte->score = (int32_t)score_to_tt(win_score, ply);
            tte->flag = TT_EXACT;
            tte->best_move_idx = (uint8_t)(m.row * BOARD_SIZE + m.col);
            return win_score;
        }

        board_place(board, m.row, m.col, player);
        state->zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;

        int score;
        if (i == 0) {
            score = -negamax(board, opp, depth - 1, ply + 1, -beta, -alpha, state);
        } else {
            score = -negamax(board, opp, depth - 1, ply + 1, -alpha - 1, -alpha, state);
            if (!state->time_up && score > alpha && score < beta) {
                score = -negamax(board, opp, depth - 1, ply + 1, -beta, -alpha, state);
            }
        }

        state->zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;
        board_undo(board, m.row, m.col);

        if (state->time_up) return 0;

        if (score > best) {
            best = score;
            best_move = m;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            if (ply < MAX_PLY && !same_move(m, state->killers[ply][0])) {
                state->killers[ply][1] = state->killers[ply][0];
                state->killers[ply][0] = m;
            }
            state->history[side_idx][m.row * BOARD_SIZE + m.col] += depth * depth * 4;
            break;
        }
    }

    TTFlag flag = TT_EXACT;
    if (best <= orig_alpha) flag = TT_UPPER;
    else if (best >= beta) flag = TT_LOWER;

    tte->key = state->zobrist;
    tte->depth = (int16_t)depth;
    tte->score = (int32_t)score_to_tt(best, ply);
    tte->flag = (uint8_t)flag;
    tte->best_move_idx = (uint8_t)(best_move.row * BOARD_SIZE + best_move.col);

    return best;
}

static void fill_policy_from_scores(
    const Move *moves,
    const int *scores,
    int count,
    float temperature,
    float *out_policy
) {
    memset(out_policy, 0, sizeof(float) * BOARD_CELLS);
    if (count <= 0) return;

    float t = temperature > 1e-4f ? temperature : 0.30f;
    int max_s = scores[0];
    for (int i = 1; i < count; ++i) {
        if (scores[i] > max_s) max_s = scores[i];
    }

    double sum = 0.0;
    double probs[32];
    for (int i = 0; i < count; ++i) {
        double scaled = ((double)(scores[i] - max_s)) / (2000.0 * (double)t);
        if (scaled < -40.0) scaled = -40.0;
        probs[i] = exp(scaled);
        sum += probs[i];
    }
    if (sum <= 0.0) {
        out_policy[moves[0].row * BOARD_SIZE + moves[0].col] = 1.0f;
        return;
    }
    for (int i = 0; i < count; ++i) {
        out_policy[moves[i].row * BOARD_SIZE + moves[i].col] = (float)(probs[i] / sum);
    }
}

SearchResult search_find_best_move(
    Board *board,
    Cell player,
    Move last_move,
    const SearchConfig *config,
    const NnueModel *nnue,
    const CnnModel *cnn
) {
    init_zobrist();

    SearchConfig cfg = config ? *config : search_default_config();
    const NnueModel *active_nnue = cfg.use_nnue ? nnue : NULL;
    const CnnModel *active_cnn = cfg.use_cnn ? cnn : NULL;
    double start_time = now_seconds();
    double budget = cfg.time_limit_sec > 0.05 ? cfg.time_limit_sec : 0.05;

    SearchResult result;
    memset(&result, 0, sizeof(result));
    result.best_move.row = -1;
    result.best_move.col = -1;
    result.stats.used_nnue = (active_nnue != NULL);

    if (player != CELL_BLACK && player != CELL_WHITE) {
        return result;
    }

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

    Move root_moves[28];
    int root_count = search_generate_candidates(board, player, 2, root_moves, 22);
    if (root_count == 0) {
        return result;
    }

    result.best_move = root_moves[0];
    result.valid = true;

    /* Layer 1 战术护栏 */
    Move guard_move;
    int guard_score = 0;
    if (check_tactical_guardrail(board, player, active_nnue, &guard_move, &guard_score)) {
        result.best_move = guard_move;
        result.root_policy[guard_move.row * BOARD_SIZE + guard_move.col] = 1.0f;
        result.stats.used_guardrail = true;
        result.stats.completed_depth = 1;
        result.stats.nodes = 1;
        result.stats.score = guard_score;
        result.stats.elapsed_sec = now_seconds() - start_time;
        return result;
    }

    /* Layer 2A：VCF 连续冲四杀棋搜索 */
    if (cfg.vcf_max_depth > 0) {
        Move vcf_move;
        uint64_t vcf_nodes = 0;
        double vcf_budget = budget * 0.25;
        if (vcf_budget > 1.5) vcf_budget = 1.5;
        if (search_find_vcf(
                board, player, cfg.vcf_max_depth, vcf_budget, &vcf_move, &vcf_nodes
            )) {
            result.best_move = vcf_move;
            result.root_policy[vcf_move.row * BOARD_SIZE + vcf_move.col] = 1.0f;
            result.stats.used_vcf = true;
            result.stats.vcf_nodes = vcf_nodes;
            result.stats.nodes = vcf_nodes;
            result.stats.completed_depth = cfg.vcf_max_depth;
            result.stats.score = EVAL_MATE_SCORE - 20;
            result.stats.elapsed_sec = now_seconds() - start_time;
            return result;
        }
        result.stats.vcf_nodes = vcf_nodes;
    }

    /* Layer 3A：CNN 根节点策略先验排序（扩展全局直觉与高概率候选点） */
    float cnn_prior[BOARD_CELLS];
    memset(cnn_prior, 0, sizeof(cnn_prior));
    if (active_cnn) {
        float cnn_in[BOARD_CELLS * CNN_INPUT_CHANNELS];
        float logits[BOARD_CELLS];
        bool legal_mask[BOARD_CELLS];
        nn_build_cnn_input(board, player, last_move, cnn_in);
        cnn_forward(active_cnn, cnn_in, logits);
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                legal_mask[r * BOARD_SIZE + c] = rules_is_legal_move(board, r, c, player);
            }
        }
        cnn_masked_softmax(logits, legal_mask, cnn_prior);
        result.stats.used_cnn = true;

        /* 把 CNN 认为概率极高的全盘合法点也补充进根节点候选（最多补 4 个） */
        for (int add = 0; add < 4 && root_count < 26; ++add) {
            int best_idx = -1;
            float best_p = 0.02f;
            for (int idx = 0; idx < BOARD_CELLS; ++idx) {
                if (!legal_mask[idx] || cnn_prior[idx] <= best_p) continue;
                int rr = idx / BOARD_SIZE;
                int cc = idx % BOARD_SIZE;
                bool exists = false;
                for (int k = 0; k < root_count; ++k) {
                    if (root_moves[k].row == rr && root_moves[k].col == cc) {
                        exists = true;
                        break;
                    }
                }
                if (!exists) {
                    best_p = cnn_prior[idx];
                    best_idx = idx;
                }
            }
            if (best_idx < 0) break;
            root_moves[root_count].row = best_idx / BOARD_SIZE;
            root_moves[root_count].col = best_idx % BOARD_SIZE;
            ++root_count;
        }

        /* 按局部战术启发 + CNN 先验排序根节点着法 */
        ScoredMove root_scored[28];
        for (int i = 0; i < root_count; ++i) {
            Move m = root_moves[i];
            int h = eval_move_heuristic(board, m.row, m.col, player);
            float p = cnn_prior[m.row * BOARD_SIZE + m.col];
            root_scored[i].move = m;
            root_scored[i].score = h + (int)(p * 6000.0f);
        }
        qsort(root_scored, (size_t)root_count, sizeof(ScoredMove), compare_scored_moves_desc);
        for (int i = 0; i < root_count; ++i) {
            root_moves[i] = root_scored[i].move;
        }
    }

    /* Layer 2B：迭代加深 PVS + 置换表 */
    TTEntry *tt = calloc(TT_SIZE, sizeof(TTEntry));
    if (!tt) {
        result.root_policy[result.best_move.row * BOARD_SIZE + result.best_move.col] = 1.0f;
        result.stats.elapsed_sec = now_seconds() - start_time;
        return result;
    }

    SearchState state;
    memset(&state, 0, sizeof(state));
    state.deadline = start_time + budget;
    state.root_player = player;
    state.zobrist = search_compute_zobrist(board, player);
    state.tt = tt;
    state.nnue = active_nnue;

    int target_depth = cfg.max_depth < 1 ? 1 : cfg.max_depth;
    Move best_move = root_moves[0];
    int best_score = 0;
    int root_scores[28];
    bool root_exact[28];
    for (int i = 0; i < root_count; ++i) {
        root_scores[i] = eval_move_heuristic(board, root_moves[i].row, root_moves[i].col, player);
        root_exact[i] = false;
    }

    int completed_depth = 0;
    Cell opp = board_opponent(player);

    for (int depth = 1; depth <= target_depth; ++depth) {
        double elapsed = now_seconds() - start_time;
        if (depth > 1 && elapsed >= budget * 0.80) {
            break;
        }

        for (int i = 1; i < root_count; ++i) {
            if (same_move(root_moves[i], best_move)) {
                Move tmp_m = root_moves[0];
                root_moves[0] = root_moves[i];
                root_moves[i] = tmp_m;
                int tmp_s = root_scores[0];
                root_scores[0] = root_scores[i];
                root_scores[i] = tmp_s;
                bool tmp_e = root_exact[0];
                root_exact[0] = root_exact[i];
                root_exact[i] = tmp_e;
                break;
            }
        }

        int iter_best_score = -EVAL_MATE_SCORE * 2;
        Move iter_best_move = root_moves[0];
        int iter_scores[28];
        bool iter_exact[28];
        int alpha = -EVAL_MATE_SCORE * 2;
        int beta = EVAL_MATE_SCORE * 2;

        for (int i = 0; i < root_count; ++i) {
            Move m = root_moves[i];
            board_place(board, m.row, m.col, player);
            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;

            int score;
            bool exact = false;
            if (i == 0) {
                score = -negamax(board, opp, depth - 1, 1, -beta, -alpha, &state);
                exact = true;
            } else {
                score = -negamax(board, opp, depth - 1, 1, -alpha - 1, -alpha, &state);
                if (!state.time_up && score > alpha && score < beta) {
                    score = -negamax(board, opp, depth - 1, 1, -beta, -alpha, &state);
                    exact = true;
                }
            }

            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;
            board_undo(board, m.row, m.col);

            if (state.time_up) {
                break;
            }

            iter_scores[i] = score;
            iter_exact[i] = exact;
            if (score > iter_best_score) {
                iter_best_score = score;
                iter_best_move = m;
            }
            if (score > alpha) {
                alpha = score;
            }
        }

        if (state.time_up) {
            break;
        }

        best_move = iter_best_move;
        best_score = iter_best_score;
        completed_depth = depth;
        for (int i = 0; i < root_count; ++i) {
            root_scores[i] = iter_scores[i];
            root_exact[i] = iter_exact[i];
        }

        if (best_score >= MATE_BOUND) {
            break;
        }
    }

    /*
     * PVS 零窗口 fail-low 着法仅返回上界，可能隐藏致命漏招。
     * 在启用 CNN 价值融合或温度采样时，用安全下界窗口对前列候选做一次精确重搜验证。
     */
    if (completed_depth >= 1
        && ((active_cnn && cfg.cnn_value_weight > 0.01f) || cfg.temperature > 0.01f)
        && best_score < MATE_BOUND && best_score > -MATE_BOUND
        && (now_seconds() - start_time) < budget * 0.88) {
        int verify_budget = 6;
        int verified_non_exact = 0;
        int safe_floor = best_score - 15000;
        int safe_ceil = best_score + 1;

        for (int i = 0; i < root_count; ++i) {
            if (root_exact[i]) {
                continue;
            }
            if (verified_non_exact >= verify_budget
                || (now_seconds() - start_time) >= budget * 0.90) {
                /* 未经精确验证的 fail-low 着法压低分数，避免被策略或融合误选 */
                if (root_scores[i] > best_score - 20000) {
                    root_scores[i] = best_score - 20000;
                }
                continue;
            }

            Move m = root_moves[i];
            state.time_up = false;
            board_place(board, m.row, m.col, player);
            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;

            int vscore = -negamax(
                board, opp, completed_depth - 1, 1, -safe_ceil, -safe_floor, &state
            );

            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;
            board_undo(board, m.row, m.col);
            ++verified_non_exact;

            if (state.time_up) {
                root_scores[i] = best_score - 20000;
                continue;
            }

            if (vscore <= safe_floor) {
                /* 确认为大漏招/明显劣招，严格压低分数并保持非精确标记 */
                root_scores[i] = best_score - 25000;
            } else {
                root_scores[i] = vscore;
                root_exact[i] = true;
                if (vscore > best_score) {
                    best_score = vscore;
                    best_move = m;
                }
            }
        }
    }

    free(tt);

    /* Layer 3B：对非杀棋局面，仅对通过精确窗口验证的顶尖候选着法进行 CNN 价值融合 */
    if (active_cnn && cfg.cnn_value_weight > 0.01f
        && best_score < MATE_BOUND && best_score > -MATE_BOUND
        && (now_seconds() - start_time) < budget * 0.92) {
        int order[28];
        for (int i = 0; i < root_count; ++i) {
            order[i] = i;
        }
        for (int i = 0; i < root_count; ++i) {
            for (int j = i + 1; j < root_count; ++j) {
                if (root_scores[order[j]] > root_scores[order[i]]) {
                    int tmp = order[i];
                    order[i] = order[j];
                    order[j] = tmp;
                }
            }
        }

        int eval_k = root_count < 6 ? root_count : 6;
        float w_cnn = cfg.cnn_value_weight;
        if (w_cnn > 0.8f) w_cnn = 0.8f;

        float best_blended = -1e9f;
        for (int k = 0; k < eval_k; ++k) {
            int idx = order[k];
            if (!root_exact[idx]) {
                continue;
            }
            if (root_scores[idx] <= -MATE_BOUND || root_scores[idx] >= MATE_BOUND) {
                continue;
            }
            /* 不挑选明显逊于最优分支超过 12000 分的坏棋 */
            if (best_score - root_scores[idx] > 12000) {
                continue;
            }

            Move m = root_moves[idx];
            board_place(board, m.row, m.col, player);
            float cnn_in[BOARD_CELLS * CNN_INPUT_CHANNELS];
            nn_build_cnn_input(board, opp, m, cnn_in);
            float opp_val = cnn_forward(active_cnn, cnn_in, NULL);
            board_undo(board, m.row, m.col);

            float my_cnn_val = -opp_val;
            float ab_norm = tanhf((float)root_scores[idx] / 25000.0f);
            float blended = (1.0f - w_cnn) * ab_norm + w_cnn * my_cnn_val;

            /* 同步修正 root_scores 以便生成融合后的策略分布 */
            root_scores[idx] = (int)(blended * 25000.0f);
            if (blended > best_blended) {
                best_blended = blended;
                best_move = m;
                best_score = root_scores[idx];
            }
        }
    }

    result.best_move = best_move;
    result.stats.completed_depth = completed_depth;
    result.stats.nodes = state.nodes + result.stats.vcf_nodes;
    result.stats.score = best_score;
    result.stats.elapsed_sec = now_seconds() - start_time;

    fill_policy_from_scores(
        root_moves, root_scores, root_count, cfg.temperature, result.root_policy
    );

    if (cfg.temperature > 0.01f && best_score < MATE_BOUND && best_score > -MATE_BOUND) {
        uint64_t rng = cfg.seed ? cfg.seed : 1ULL;
        uint64_t rval = splitmix64(&rng);
        double u = (double)(rval & 0xFFFFFFFFULL) / 4294967296.0;
        double acc = 0.0;
        for (int i = 0; i < root_count; ++i) {
            int idx = root_moves[i].row * BOARD_SIZE + root_moves[i].col;
            acc += (double)result.root_policy[idx];
            if (u <= acc) {
                result.best_move = root_moves[i];
                break;
            }
        }
    }

    return result;
}

/* ==================== V3 Pattern-NNUE + 持久 TT + LMR ==================== */

#define PNNUE_TT_SIZE (1U << 19)
#define PNNUE_TT_MASK (PNNUE_TT_SIZE - 1U)

typedef struct {
    uint64_t key;
    int32_t score;
    int16_t depth;
    uint8_t flag;
    uint8_t best_move_idx;
    uint32_t gen;
} PnnueTTEntry;

typedef struct {
    double deadline;
    bool time_up;
    uint64_t nodes;
    Cell root_player;
    uint64_t zobrist;
    uint32_t tt_gen;
    PnnueTTEntry *tt;
    const PatternNnueModel *pnnue;
    PnnueAcc *acc;
    Move killers[MAX_PLY][2];
    int history[2][BOARD_CELLS];
} PnnueSearchState;

static PnnueTTEntry *g_pnnue_tt = NULL;
static uint32_t g_pnnue_tt_gen = 0;

static PnnueTTEntry *acquire_pnnue_tt(uint32_t *out_gen) {
    if (!g_pnnue_tt) {
        g_pnnue_tt = (PnnueTTEntry *)calloc(PNNUE_TT_SIZE, sizeof(PnnueTTEntry));
        g_pnnue_tt_gen = 1;
    } else {
        ++g_pnnue_tt_gen;
        if (g_pnnue_tt_gen == 0) {
            memset(g_pnnue_tt, 0, sizeof(PnnueTTEntry) * PNNUE_TT_SIZE);
            g_pnnue_tt_gen = 1;
        }
    }
    *out_gen = g_pnnue_tt_gen;
    return g_pnnue_tt;
}

static int generate_pnnue_moves(
    Board *board,
    Cell player,
    int ply,
    Move tt_move,
    const PnnueSearchState *state,
    Move *out_moves,
    int max_moves
) {
    ScoredMove scored[BOARD_CELLS];
    int count = 0;
    int side_idx = (player == CELL_BLACK) ? 0 : 1;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (!board_is_empty(board, r, c)) continue;
            if (!has_neighbor_within(board, r, c, 2)) continue;
            if (!rules_is_legal_move(board, r, c, player)) continue;

            Move m = {r, c};
            float logit = pnnue_eval_policy_cell(state->acc, state->pnnue, r, c, player);
            int h = eval_move_heuristic(board, r, c, player);
            int s = (int)(logit * 1200.0f) + (h / 2);

            if (same_move(m, tt_move)) {
                s += 5000000;
            } else if (ply < MAX_PLY && same_move(m, state->killers[ply][0])) {
                s += 250000;
            } else if (ply < MAX_PLY && same_move(m, state->killers[ply][1])) {
                s += 180000;
            }
            s += state->history[side_idx][r * BOARD_SIZE + c];

            scored[count].move = m;
            scored[count].score = s;
            ++count;
        }
    }

    if (count == 0) {
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                if (rules_is_legal_move(board, r, c, player)) {
                    scored[count].move.row = r;
                    scored[count].move.col = c;
                    scored[count].score = (int)(
                        pnnue_eval_policy_cell(state->acc, state->pnnue, r, c, player) * 1200.0f
                    ) + (eval_move_heuristic(board, r, c, player) / 2);
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

static int negamax_pnnue(
    Board *board,
    Cell player,
    int depth,
    int ply,
    int alpha,
    int beta,
    PnnueSearchState *state
) {
    ++state->nodes;
    if ((state->nodes & 1023ULL) == 0ULL && now_seconds() >= state->deadline) {
        state->time_up = true;
        return 0;
    }

    if (board_is_full(board)) {
        return 0;
    }

    int orig_alpha = alpha;
    PnnueTTEntry *tte = &state->tt[state->zobrist & PNNUE_TT_MASK];
    Move tt_move = {-1, -1};
    if (tte->flag != TT_EMPTY && tte->key == state->zobrist) {
        int idx = tte->best_move_idx;
        if (idx < BOARD_CELLS) {
            tt_move.row = idx / BOARD_SIZE;
            tt_move.col = idx % BOARD_SIZE;
        }
        if (tte->depth >= depth) {
            int tt_score = score_from_tt(tte->score, ply);
            if (tte->flag == TT_EXACT) return tt_score;
            if (tte->flag == TT_LOWER && tt_score > alpha) alpha = tt_score;
            else if (tte->flag == TT_UPPER && tt_score < beta) beta = tt_score;
            if (alpha >= beta) return tt_score;
        }
    }

    if (depth <= 0) {
        int lin_score = eval_position(board, player);
        if (lin_score >= MATE_BOUND || lin_score <= -MATE_BOUND) {
            return lin_score;
        }
        float val = pnnue_eval_value(state->acc, state->pnnue, player);
        int nn_score = (int)(val * 22000.0f);
        int blended = nn_score + (lin_score / 2);
        if (blended > EVAL_MAX_STATIC_SCORE) blended = EVAL_MAX_STATIC_SCORE;
        if (blended < -EVAL_MAX_STATIC_SCORE) blended = -EVAL_MAX_STATIC_SCORE;
        return blended;
    }

    Move candidates[24];
    int limit = depth >= 3 ? 20 : 16;
    int cand_count = generate_pnnue_moves(
        board, player, ply, tt_move, state, candidates, limit
    );
    if (cand_count == 0) {
        return -(EVAL_MATE_SCORE - ply);
    }

    int best = -EVAL_MATE_SCORE * 2;
    Move best_move = candidates[0];
    int side_idx = (player == CELL_BLACK) ? 0 : 1;
    Cell opp = board_opponent(player);

    for (int i = 0; i < cand_count; ++i) {
        Move m = candidates[i];

        if (rules_is_winning_move(board, m.row, m.col, player)) {
            int win_score = EVAL_MATE_SCORE - ply - 1;
            tte->key = state->zobrist;
            tte->depth = (int16_t)depth;
            tte->score = (int32_t)score_to_tt(win_score, ply);
            tte->flag = TT_EXACT;
            tte->best_move_idx = (uint8_t)(m.row * BOARD_SIZE + m.col);
            tte->gen = state->tt_gen;
            return win_score;
        }

        board_place(board, m.row, m.col, player);
        pnnue_acc_update_cell(state->acc, state->pnnue, m.row, m.col, player);
        state->zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;

        int score;
        if (i == 0) {
            score = -negamax_pnnue(board, opp, depth - 1, ply + 1, -beta, -alpha, state);
        } else {
            int reduction = 0;
            if (depth >= 4 && i >= 6
                && eval_move_heuristic(board, m.row, m.col, player) < 4000) {
                reduction = (depth >= 6 && i >= 10) ? 2 : 1;
            }
            int search_d = depth - 1 - reduction;
            if (search_d < 0) search_d = 0;

            score = -negamax_pnnue(
                board, opp, search_d, ply + 1, -alpha - 1, -alpha, state
            );
            if (!state->time_up && reduction > 0 && score > alpha) {
                score = -negamax_pnnue(
                    board, opp, depth - 1, ply + 1, -alpha - 1, -alpha, state
                );
            }
            if (!state->time_up && score > alpha && score < beta) {
                score = -negamax_pnnue(
                    board, opp, depth - 1, ply + 1, -beta, -alpha, state
                );
            }
        }

        state->zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;
        pnnue_acc_update_cell(state->acc, state->pnnue, m.row, m.col, CELL_EMPTY);
        board_undo(board, m.row, m.col);

        if (state->time_up) return 0;

        if (score > best) {
            best = score;
            best_move = m;
        }
        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            if (ply < MAX_PLY && !same_move(m, state->killers[ply][0])) {
                state->killers[ply][1] = state->killers[ply][0];
                state->killers[ply][0] = m;
            }
            state->history[side_idx][m.row * BOARD_SIZE + m.col] += depth * depth * 4;
            break;
        }
    }

    if (tte->flag == TT_EMPTY || tte->gen != state->tt_gen ||
        tte->key == state->zobrist || depth >= tte->depth) {
        TTFlag flag = TT_EXACT;
        if (best <= orig_alpha) flag = TT_UPPER;
        else if (best >= beta) flag = TT_LOWER;

        tte->key = state->zobrist;
        tte->depth = (int16_t)depth;
        tte->score = (int32_t)score_to_tt(best, ply);
        tte->flag = (uint8_t)flag;
        tte->best_move_idx = (uint8_t)(best_move.row * BOARD_SIZE + best_move.col);
        tte->gen = state->tt_gen;
    }

    return best;
}

SearchResult search_find_best_move_pnnue(
    Board *board,
    Cell player,
    Move last_move,
    const SearchConfig *config,
    const PatternNnueModel *pnnue
) {
    if (!pnnue) {
        return search_find_best_move(board, player, last_move, config, NULL, NULL);
    }

    init_zobrist();
    pnnue_init_tables();

    SearchConfig cfg = config ? *config : search_default_config();
    double start_time = now_seconds();
    double budget = cfg.time_limit_sec > 0.05 ? cfg.time_limit_sec : 0.05;

    SearchResult result;
    memset(&result, 0, sizeof(result));
    result.best_move.row = -1;
    result.best_move.col = -1;
    result.stats.used_nnue = true;

    if (player != CELL_BLACK && player != CELL_WHITE) {
        return result;
    }

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

    /* Layer 1 战术护栏 */
    Move guard_move;
    int guard_score = 0;
    if (check_tactical_guardrail(board, player, NULL, &guard_move, &guard_score)) {
        result.best_move = guard_move;
        result.root_policy[guard_move.row * BOARD_SIZE + guard_move.col] = 1.0f;
        result.valid = true;
        result.stats.used_guardrail = true;
        result.stats.completed_depth = 1;
        result.stats.nodes = 1;
        result.stats.score = guard_score;
        result.stats.elapsed_sec = now_seconds() - start_time;
        return result;
    }

    /* Layer 2A：VCF 连续冲四杀棋搜索 */
    if (cfg.vcf_max_depth > 0) {
        Move vcf_move;
        uint64_t vcf_nodes = 0;
        double vcf_budget = budget * 0.25;
        if (vcf_budget > 1.5) vcf_budget = 1.5;
        if (search_find_vcf(
                board, player, cfg.vcf_max_depth, vcf_budget, &vcf_move, &vcf_nodes
            )) {
            result.best_move = vcf_move;
            result.root_policy[vcf_move.row * BOARD_SIZE + vcf_move.col] = 1.0f;
            result.valid = true;
            result.stats.used_vcf = true;
            result.stats.vcf_nodes = vcf_nodes;
            result.stats.nodes = vcf_nodes;
            result.stats.completed_depth = cfg.vcf_max_depth;
            result.stats.score = EVAL_MATE_SCORE - 20;
            result.stats.elapsed_sec = now_seconds() - start_time;
            return result;
        }
        result.stats.vcf_nodes = vcf_nodes;
    }

    PnnueAcc *acc = (PnnueAcc *)calloc(1, sizeof(PnnueAcc));
    if (!acc) {
        return result;
    }
    pnnue_acc_init(acc, pnnue, board);

    uint32_t tt_gen = 1;
    PnnueTTEntry *tt = acquire_pnnue_tt(&tt_gen);
    if (!tt) {
        free(acc);
        return result;
    }

    uint64_t model_seed = (uint64_t)(uintptr_t)pnnue;
    uint64_t model_salt = splitmix64(&model_seed);

    PnnueSearchState state;
    memset(&state, 0, sizeof(state));
    state.deadline = start_time + budget;
    state.root_player = player;
    state.zobrist = search_compute_zobrist(board, player) ^ model_salt;
    state.tt_gen = tt_gen;
    state.tt = tt;
    state.pnnue = pnnue;
    state.acc = acc;

    Move root_moves[28];
    Move no_tt = {-1, -1};
    int root_count = generate_pnnue_moves(board, player, 0, no_tt, &state, root_moves, 24);
    if (root_count == 0) {
        free(acc);
        return result;
    }

    result.best_move = root_moves[0];
    result.valid = true;

    int target_depth = cfg.max_depth < 1 ? 1 : cfg.max_depth;
    Move best_move = root_moves[0];
    int best_score = 0;
    int root_scores[28];
    bool root_exact[28];
    for (int i = 0; i < root_count; ++i) {
        float lg = pnnue_eval_policy_cell(acc, pnnue, root_moves[i].row, root_moves[i].col, player);
        root_scores[i] = (int)(lg * 1000.0f);
        root_exact[i] = false;
    }

    int completed_depth = 0;
    Cell opp = board_opponent(player);

    for (int depth = 1; depth <= target_depth; ++depth) {
        double elapsed = now_seconds() - start_time;
        if (depth > 1 && elapsed >= budget * 0.80) {
            break;
        }

        for (int i = 1; i < root_count; ++i) {
            if (same_move(root_moves[i], best_move)) {
                Move tmp_m = root_moves[0];
                root_moves[0] = root_moves[i];
                root_moves[i] = tmp_m;
                int tmp_s = root_scores[0];
                root_scores[0] = root_scores[i];
                root_scores[i] = tmp_s;
                bool tmp_e = root_exact[0];
                root_exact[0] = root_exact[i];
                root_exact[i] = tmp_e;
                break;
            }
        }

        int iter_best_score = -EVAL_MATE_SCORE * 2;
        Move iter_best_move = root_moves[0];
        int iter_scores[28];
        bool iter_exact[28];
        int alpha = -EVAL_MATE_SCORE * 2;
        int beta = EVAL_MATE_SCORE * 2;

        for (int i = 0; i < root_count; ++i) {
            Move m = root_moves[i];
            board_place(board, m.row, m.col, player);
            pnnue_acc_update_cell(acc, pnnue, m.row, m.col, player);
            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;

            int score;
            bool exact = false;
            if (i == 0) {
                score = -negamax_pnnue(board, opp, depth - 1, 1, -beta, -alpha, &state);
                exact = true;
            } else {
                score = -negamax_pnnue(board, opp, depth - 1, 1, -alpha - 1, -alpha, &state);
                if (!state.time_up && score > alpha && score < beta) {
                    score = -negamax_pnnue(board, opp, depth - 1, 1, -beta, -alpha, &state);
                    exact = true;
                }
            }

            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;
            pnnue_acc_update_cell(acc, pnnue, m.row, m.col, CELL_EMPTY);
            board_undo(board, m.row, m.col);

            if (state.time_up) {
                break;
            }

            iter_scores[i] = score;
            iter_exact[i] = exact;
            if (score > iter_best_score) {
                iter_best_score = score;
                iter_best_move = m;
            }
            if (score > alpha) {
                alpha = score;
            }
        }

        if (state.time_up) {
            break;
        }

        best_move = iter_best_move;
        best_score = iter_best_score;
        completed_depth = depth;
        for (int i = 0; i < root_count; ++i) {
            root_scores[i] = iter_scores[i];
            root_exact[i] = iter_exact[i];
        }

        if (best_score >= MATE_BOUND) {
            break;
        }
    }

    /* PVS 根节点 fail-low 精确窗口复核，防止温度采样与策略目标混入盲区坏棋 */
    if (completed_depth >= 1
        && cfg.temperature > 0.01f
        && best_score < MATE_BOUND && best_score > -MATE_BOUND
        && (now_seconds() - start_time) < budget * 0.88) {
        int verify_budget = 6;
        int verified_non_exact = 0;
        int safe_floor = best_score - 15000;
        int safe_ceil = best_score + 1;

        for (int i = 0; i < root_count; ++i) {
            if (root_exact[i]) {
                continue;
            }
            if (verified_non_exact >= verify_budget
                || (now_seconds() - start_time) >= budget * 0.90) {
                if (root_scores[i] > best_score - 20000) {
                    root_scores[i] = best_score - 20000;
                }
                continue;
            }

            Move m = root_moves[i];
            state.time_up = false;
            board_place(board, m.row, m.col, player);
            pnnue_acc_update_cell(acc, pnnue, m.row, m.col, player);
            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;

            int vscore = -negamax_pnnue(
                board, opp, completed_depth - 1, 1, -safe_ceil, -safe_floor, &state
            );

            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;
            pnnue_acc_update_cell(acc, pnnue, m.row, m.col, CELL_EMPTY);
            board_undo(board, m.row, m.col);
            ++verified_non_exact;

            if (state.time_up) {
                root_scores[i] = best_score - 20000;
                continue;
            }

            if (vscore <= safe_floor) {
                root_scores[i] = best_score - 25000;
            } else {
                root_scores[i] = vscore;
                root_exact[i] = true;
                if (vscore > best_score) {
                    best_score = vscore;
                    best_move = m;
                }
            }
        }
    } else if (completed_depth >= 1) {
        for (int i = 0; i < root_count; ++i) {
            if (!root_exact[i] && root_scores[i] > best_score - 20000) {
                root_scores[i] = best_score - 20000;
            }
        }
    }

    free(acc);

    result.best_move = best_move;
    result.stats.completed_depth = completed_depth;
    result.stats.nodes = state.nodes + result.stats.vcf_nodes;
    result.stats.score = best_score;
    result.stats.elapsed_sec = now_seconds() - start_time;

    fill_policy_from_scores(
        root_moves, root_scores, root_count, cfg.temperature, result.root_policy
    );

    if (cfg.temperature > 0.01f && best_score < MATE_BOUND && best_score > -MATE_BOUND) {
        uint64_t rng = cfg.seed ? cfg.seed : 1ULL;
        uint64_t rval = splitmix64(&rng);
        double u = (double)(rval & 0xFFFFFFFFULL) / 4294967296.0;
        double acc_p = 0.0;
        for (int i = 0; i < root_count; ++i) {
            int idx = root_moves[i].row * BOARD_SIZE + root_moves[i].col;
            acc_p += (double)result.root_policy[idx];
            if (u <= acc_p) {
                result.best_move = root_moves[i];
                break;
            }
        }
    }

    return result;
}
