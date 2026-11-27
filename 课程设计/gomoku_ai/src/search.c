#include "search.h"

#include "eval.h"
#include "rules.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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
            if (!board_is_empty(board, nr, nc)) {
                continue;
            }
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

    /* 1. 立即成五必胜 */
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
        if (*time_up) {
            return false;
        }
        Move m = cands[i];
        board_place(board, m.row, m.col, attacker);

        Move win_pts[8];
        int wp_count = collect_four_winning_points(
            board, m.row, m.col, attacker, win_pts, 8
        );

        if (wp_count >= 2) {
            /* 形成活四或双冲四，无法单手解 */
            board_undo(board, m.row, m.col);
            if (win_move) {
                *win_move = m;
            }
            return true;
        }

        if (wp_count == 1) {
            Move def = win_pts[0];
            bool def_wins = rules_is_winning_move(board, def.row, def.col, defender);
            if (!def_wins) {
                if (!rules_is_legal_move(board, def.row, def.col, defender)) {
                    /* 防守方被迫下禁手点（黑方），进攻方直接胜 */
                    board_undo(board, m.row, m.col);
                    if (win_move) {
                        *win_move = m;
                    }
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
                    if (win_move) {
                        *win_move = m;
                    }
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
    Move *out_move,
    int *out_score
) {
    /* 1. 己方一步成五 */
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

    /* 2. 对方存在唯一成五点，必须立即封堵 */
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
    /* 保留超过裁剪上限的强威胁点（至少活三/冲四等级） */
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
            if (tte->flag == TT_EXACT) {
                return tt_score;
            }
            if (tte->flag == TT_LOWER && tt_score > alpha) {
                alpha = tt_score;
            } else if (tte->flag == TT_UPPER && tt_score < beta) {
                beta = tt_score;
            }
            if (alpha >= beta) {
                return tt_score;
            }
        }
    }

    if (depth <= 0) {
        return eval_position(board, player);
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

        if (state->time_up) {
            return 0;
        }

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
    if (best <= orig_alpha) {
        flag = TT_UPPER;
    } else if (best >= beta) {
        flag = TT_LOWER;
    }
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
    if (count <= 0) {
        return;
    }
    if (temperature <= 1e-4f) {
        temperature = 0.25f;
    }
    int max_s = scores[0];
    for (int i = 1; i < count; ++i) {
        if (scores[i] > max_s) max_s = scores[i];
    }

    double sum = 0.0;
    double probs[32];
    for (int i = 0; i < count; ++i) {
        double scaled = ((double)(scores[i] - max_s)) / (1500.0 * (double)temperature);
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
    (void)last_move;
    (void)nnue;
    (void)cnn;

    init_zobrist();

    SearchConfig cfg = config ? *config : search_default_config();
    double start_time = now_seconds();
    double budget = cfg.time_limit_sec > 0.05 ? cfg.time_limit_sec : 0.05;

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

    int target_depth = cfg.max_depth < 1 ? 1 : cfg.max_depth;
    Move best_move = root_moves[0];
    int best_score = 0;
    int root_scores[28];
    for (int i = 0; i < root_count; ++i) {
        root_scores[i] = eval_move_heuristic(board, root_moves[i].row, root_moves[i].col, player);
    }

    int completed_depth = 0;
    Cell opp = board_opponent(player);

    for (int depth = 1; depth <= target_depth; ++depth) {
        double elapsed = now_seconds() - start_time;
        if (depth > 1 && elapsed >= budget * 0.80) {
            break;
        }

        /* 将上轮最佳着法移至根节点首位 */
        for (int i = 1; i < root_count; ++i) {
            if (same_move(root_moves[i], best_move)) {
                Move tmp_m = root_moves[0];
                root_moves[0] = root_moves[i];
                root_moves[i] = tmp_m;
                int tmp_s = root_scores[0];
                root_scores[0] = root_scores[i];
                root_scores[i] = tmp_s;
                break;
            }
        }

        int iter_best_score = -EVAL_MATE_SCORE * 2;
        Move iter_best_move = root_moves[0];
        int iter_scores[28];
        int alpha = -EVAL_MATE_SCORE * 2;
        int beta = EVAL_MATE_SCORE * 2;

        for (int i = 0; i < root_count; ++i) {
            Move m = root_moves[i];
            board_place(board, m.row, m.col, player);
            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;

            int score;
            if (i == 0) {
                score = -negamax(board, opp, depth - 1, 1, -beta, -alpha, &state);
            } else {
                score = -negamax(board, opp, depth - 1, 1, -alpha - 1, -alpha, &state);
                if (!state.time_up && score > alpha && score < beta) {
                    score = -negamax(board, opp, depth - 1, 1, -beta, -alpha, &state);
                }
            }

            state.zobrist ^= g_zobrist_stone[m.row][m.col][player] ^ g_zobrist_side;
            board_undo(board, m.row, m.col);

            if (state.time_up) {
                break;
            }

            iter_scores[i] = score;
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
        }

        if (best_score >= MATE_BOUND) {
            break;
        }
    }

    free(tt);

    result.best_move = best_move;
    result.stats.completed_depth = completed_depth;
    result.stats.nodes = state.nodes + result.stats.vcf_nodes;
    result.stats.score = best_score;
    result.stats.elapsed_sec = now_seconds() - start_time;

    fill_policy_from_scores(
        root_moves, root_scores, root_count, cfg.temperature, result.root_policy
    );

    /* 若需要温度探索采样（自博弈多样性） */
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
