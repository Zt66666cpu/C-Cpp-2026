#include "eval.h"

#include <stddef.h>
#include <string.h>

typedef struct {
    int five;
    int open_four;
    int rush_four;
    int open_three;
    int sleep_three;
    int open_two;
    int sleep_two;
    int max_run;
} PatternCounts;

static int clamp_int(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

/* 对单一序列（元素仅含 0=空位、1=己方子、2=阻挡/墙/对方子）统计棋型 */
static void scan_binary_line(
    const int *line,
    int len,
    PatternCounts *counts
) {
    int start = 0;

    while (start < len) {
        if (line[start] == 2) {
            ++start;
            continue;
        }

        int end = start;
        while (end < len && line[end] != 2) {
            ++end;
        }

        int seg_len = end - start;
        if (seg_len < 5) {
            start = end;
            continue;
        }

        const int *seg = line + start;
        bool covered[19];
        for (int i = 0; i < seg_len; ++i) {
            covered[i] = false;
        }

        /* 1. 连续五连及以上 */
        int i = 0;
        while (i < seg_len) {
            if (seg[i] != 1) {
                ++i;
                continue;
            }
            int run_end = i;
            while (run_end < seg_len && seg[run_end] == 1) {
                ++run_end;
            }
            int run_len = run_end - i;
            if (run_len > counts->max_run) {
                counts->max_run = run_len;
            }
            if (run_len >= 5) {
                ++counts->five;
                for (int k = i; k < run_end; ++k) {
                    covered[k] = true;
                }
            }
            i = run_end;
        }

        /* 2. 活四：0 1 1 1 1 0 */
        for (int j = 0; j <= seg_len - 6; ++j) {
            if (seg[j] == 0
                && seg[j + 1] == 1
                && seg[j + 2] == 1
                && seg[j + 3] == 1
                && seg[j + 4] == 1
                && seg[j + 5] == 0
                && !covered[j + 1]
                && !covered[j + 2]
                && !covered[j + 3]
                && !covered[j + 4]) {
                ++counts->open_four;
                for (int k = j + 1; k <= j + 4; ++k) {
                    covered[k] = true;
                }
            }
        }

        /* 3. 冲四（含跳冲四）：长度为 5 的无阻挡窗口中有 4 子 1 空 */
        for (int j = 0; j <= seg_len - 5; ++j) {
            int ones = 0;
            bool all_covered = true;
            for (int k = 0; k < 5; ++k) {
                if (seg[j + k] == 1) {
                    ++ones;
                    if (!covered[j + k]) {
                        all_covered = false;
                    }
                }
            }
            if (ones == 4 && !all_covered) {
                ++counts->rush_four;
                for (int k = 0; k < 5; ++k) {
                    if (seg[j + k] == 1) {
                        covered[j + k] = true;
                    }
                }
            }
        }

        /* 4. 活三：连活三（两侧留空且至少一侧有两空）与跳活三 */
        for (int j = 0; j <= seg_len - 5; ++j) {
            if (seg[j] == 0
                && seg[j + 1] == 1
                && seg[j + 2] == 1
                && seg[j + 3] == 1
                && seg[j + 4] == 0
                && ((j > 0 && seg[j - 1] == 0)
                    || (j + 5 < seg_len && seg[j + 5] == 0))
                && !covered[j + 1]
                && !covered[j + 2]
                && !covered[j + 3]) {
                ++counts->open_three;
                covered[j + 1] = true;
                covered[j + 2] = true;
                covered[j + 3] = true;
            }
        }

        for (int j = 0; j <= seg_len - 6; ++j) {
            bool pattern_a = seg[j] == 0 && seg[j + 1] == 1 && seg[j + 2] == 0
                && seg[j + 3] == 1 && seg[j + 4] == 1 && seg[j + 5] == 0;
            bool pattern_b = seg[j] == 0 && seg[j + 1] == 1 && seg[j + 2] == 1
                && seg[j + 3] == 0 && seg[j + 4] == 1 && seg[j + 5] == 0;
            if ((pattern_a || pattern_b)
                && (j == 0 || seg[j - 1] == 0)
                && (j + 6 == seg_len || seg[j + 6] == 0)) {
                bool any_covered = false;
                for (int k = 1; k <= 4; ++k) {
                    if (seg[j + k] == 1 && covered[j + k]) {
                        any_covered = true;
                    }
                }
                if (!any_covered) {
                    ++counts->open_three;
                    for (int k = 1; k <= 4; ++k) {
                        if (seg[j + k] == 1) {
                            covered[j + k] = true;
                        }
                    }
                }
            }
        }

        /* 5. 眠三：5 格窗口内 3 子 2 空且未被更强棋型覆盖 */
        for (int j = 0; j <= seg_len - 5; ++j) {
            int ones = 0;
            bool any_covered = false;
            for (int k = 0; k < 5; ++k) {
                if (seg[j + k] == 1) {
                    ++ones;
                    if (covered[j + k]) {
                        any_covered = true;
                    }
                }
            }
            if (ones == 3 && !any_covered) {
                ++counts->sleep_three;
                for (int k = 0; k < 5; ++k) {
                    if (seg[j + k] == 1) {
                        covered[j + k] = true;
                    }
                }
            }
        }

        /* 6. 活二：连活二与跳活二 */
        for (int j = 0; j <= seg_len - 4; ++j) {
            if (seg[j] == 0
                && seg[j + 1] == 1
                && seg[j + 2] == 1
                && seg[j + 3] == 0
                && ((j > 0 && seg[j - 1] == 0)
                    || (j + 4 < seg_len && seg[j + 4] == 0))
                && !covered[j + 1]
                && !covered[j + 2]) {
                ++counts->open_two;
                covered[j + 1] = true;
                covered[j + 2] = true;
            }
        }

        for (int j = 0; j <= seg_len - 5; ++j) {
            if (seg[j] == 0
                && seg[j + 1] == 1
                && seg[j + 2] == 0
                && seg[j + 3] == 1
                && seg[j + 4] == 0
                && !covered[j + 1]
                && !covered[j + 3]) {
                ++counts->open_two;
                covered[j + 1] = true;
                covered[j + 3] = true;
            }
        }

        /* 7. 眠二：5 格窗口内 2 子 3 空且未被覆盖 */
        for (int j = 0; j <= seg_len - 5; ++j) {
            int ones = 0;
            bool any_covered = false;
            for (int k = 0; k < 5; ++k) {
                if (seg[j + k] == 1) {
                    ++ones;
                    if (covered[j + k]) {
                        any_covered = true;
                    }
                }
            }
            if (ones == 2 && !any_covered) {
                ++counts->sleep_two;
                for (int k = 0; k < 5; ++k) {
                    if (seg[j + k] == 1) {
                        covered[j + k] = true;
                    }
                }
            }
        }

        start = end;
    }
}

static void extract_board_line_and_scan(
    const Board *board,
    int r0,
    int c0,
    int dr,
    int dc,
    int len,
    Cell my,
    Cell opp,
    PatternCounts *my_counts,
    PatternCounts *opp_counts
) {
    int my_line[BOARD_SIZE + 2];
    int opp_line[BOARD_SIZE + 2];

    my_line[0] = 2;
    opp_line[0] = 2;

    for (int i = 0; i < len; ++i) {
        Cell cell = board->cells[r0 + i * dr][c0 + i * dc];
        if (cell == my) {
            my_line[i + 1] = 1;
            opp_line[i + 1] = 2;
        } else if (cell == opp) {
            my_line[i + 1] = 2;
            opp_line[i + 1] = 1;
        } else {
            my_line[i + 1] = 0;
            opp_line[i + 1] = 0;
        }
    }

    my_line[len + 1] = 2;
    opp_line[len + 1] = 2;

    scan_binary_line(my_line, len + 2, my_counts);
    scan_binary_line(opp_line, len + 2, opp_counts);
}

void eval_extract_features(
    const Board *board,
    Cell side_to_move,
    EvalFeatures *out
) {
    memset(out, 0, sizeof(*out));

    Cell my = (side_to_move == CELL_BLACK || side_to_move == CELL_WHITE)
        ? side_to_move
        : CELL_BLACK;
    Cell opp = board_opponent(my);

    PatternCounts my_cnt = {0};
    PatternCounts opp_cnt = {0};

    /* 15 行与 15 列 */
    for (int i = 0; i < BOARD_SIZE; ++i) {
        extract_board_line_and_scan(
            board, i, 0, 0, 1, BOARD_SIZE, my, opp, &my_cnt, &opp_cnt
        );
        extract_board_line_and_scan(
            board, 0, i, 1, 0, BOARD_SIZE, my, opp, &my_cnt, &opp_cnt
        );
    }

    /* 主斜线 (1, 1)，仅扫描长度 >= 5 的 21 条线 */
    for (int r = 0; r <= BOARD_SIZE - 5; ++r) {
        extract_board_line_and_scan(
            board, r, 0, 1, 1, BOARD_SIZE - r, my, opp, &my_cnt, &opp_cnt
        );
    }
    for (int c = 1; c <= BOARD_SIZE - 5; ++c) {
        extract_board_line_and_scan(
            board, 0, c, 1, 1, BOARD_SIZE - c, my, opp, &my_cnt, &opp_cnt
        );
    }

    /* 副斜线 (1, -1)，仅扫描长度 >= 5 的 21 条线 */
    for (int r = 0; r <= BOARD_SIZE - 5; ++r) {
        extract_board_line_and_scan(
            board, r, BOARD_SIZE - 1, 1, -1, BOARD_SIZE - r,
            my, opp, &my_cnt, &opp_cnt
        );
    }
    for (int c = 4; c <= BOARD_SIZE - 2; ++c) {
        extract_board_line_and_scan(
            board, 0, c, 1, -1, c + 1, my, opp, &my_cnt, &opp_cnt
        );
    }

    /* 空间控制、局部连通度与贴身压力 */
    float my_center = 0.0f;
    float opp_center = 0.0f;
    float my_adj = 0.0f;
    float opp_adj = 0.0f;
    float my_press = 0.0f;
    float opp_press = 0.0f;
    int my_stones = 0;
    int opp_stones = 0;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            Cell cell = board->cells[r][c];
            if (cell == CELL_EMPTY) {
                continue;
            }
            int dr = r - 7;
            int dc = c - 7;
            if (dr < 0) dr = -dr;
            if (dc < 0) dc = -dc;
            int dist = dr > dc ? dr : dc;
            float cw = (float)(8 - dist) / 7.0f;

            int same_nbr = 0;
            int opp_nbr = 0;
            for (int kr = -1; kr <= 1; ++kr) {
                for (int kc = -1; kc <= 1; ++kc) {
                    if (kr == 0 && kc == 0) {
                        continue;
                    }
                    int nr = r + kr;
                    int nc = c + kc;
                    if (!board_is_inside(nr, nc)) {
                        continue;
                    }
                    Cell nb = board->cells[nr][nc];
                    if (nb == cell) {
                        ++same_nbr;
                    } else if (nb != CELL_EMPTY) {
                        ++opp_nbr;
                    }
                }
            }

            if (cell == my) {
                ++my_stones;
                my_center += cw;
                my_adj += (float)same_nbr;
                my_press += (float)opp_nbr;
            } else {
                ++opp_stones;
                opp_center += cw;
                opp_adj += (float)same_nbr;
                opp_press += (float)opp_nbr;
            }
        }
    }

    float *f = out->values;
    f[0] = (float)my_cnt.five;
    f[1] = (float)opp_cnt.five;
    f[2] = (float)my_cnt.open_four;
    f[3] = (float)opp_cnt.open_four;
    f[4] = (float)my_cnt.rush_four;
    f[5] = (float)opp_cnt.rush_four;
    f[6] = (float)my_cnt.open_three;
    f[7] = (float)opp_cnt.open_three;
    f[8] = (float)my_cnt.sleep_three;
    f[9] = (float)opp_cnt.sleep_three;
    f[10] = (float)my_cnt.open_two;
    f[11] = (float)opp_cnt.open_two;
    f[12] = (float)my_cnt.sleep_two;
    f[13] = (float)opp_cnt.sleep_two;

    f[14] = (my_cnt.open_four >= 1 || my_cnt.rush_four >= 2
             || (my_cnt.rush_four >= 1 && my_cnt.open_three >= 1))
        ? 1.0f : 0.0f;
    f[15] = (opp_cnt.open_four >= 1 || opp_cnt.rush_four >= 2
             || (opp_cnt.rush_four >= 1 && opp_cnt.open_three >= 1))
        ? 1.0f : 0.0f;
    f[16] = (my_cnt.open_three >= 2) ? 1.0f : 0.0f;
    f[17] = (opp_cnt.open_three >= 2) ? 1.0f : 0.0f;

    f[18] = (float)(my_cnt.open_four + my_cnt.rush_four + my_cnt.open_three);
    f[19] = (float)(opp_cnt.open_four + opp_cnt.rush_four + opp_cnt.open_three);

    f[20] = my_center / 16.0f;
    f[21] = opp_center / 16.0f;
    f[22] = my_adj / 32.0f;
    f[23] = opp_adj / 32.0f;
    f[24] = my_press / 24.0f;
    f[25] = opp_press / 24.0f;
    f[26] = (float)my_stones / 112.5f;
    f[27] = (float)opp_stones / 112.5f;
    f[28] = (my == CELL_BLACK) ? 1.0f : 0.0f;
    f[29] = (float)(my_stones + opp_stones) / (float)BOARD_CELLS;
    f[30] = (float)(my_cnt.max_run > 5 ? 5 : my_cnt.max_run) / 5.0f;
    f[31] = (float)(opp_cnt.max_run > 5 ? 5 : opp_cnt.max_run) / 5.0f;
}

int eval_linear_score(const EvalFeatures *features) {
    /* 当前行棋方视角：下一手轮到己方，己方同级棋型赋予略高于对方的先手系数 */
    const float *f = features->values;
    float score = 0.0f;

    score += f[0] * 120000.0f - f[1] * 130000.0f; /* five */
    score += f[2] * 36000.0f  - f[3] * 48000.0f;  /* open_four */
    score += f[4] * 14000.0f  - f[5] * 12000.0f;  /* rush_four (己方轮走权加成) */
    score += f[6] * 5600.0f   - f[7] * 7200.0f;   /* open_three */
    score += f[8] * 900.0f    - f[9] * 750.0f;    /* sleep_three */
    score += f[10] * 650.0f   - f[11] * 600.0f;   /* open_two */
    score += f[12] * 140.0f   - f[13] * 120.0f;   /* sleep_two */

    score += f[14] * 24000.0f - f[15] * 28000.0f; /* four_three combo */
    score += f[16] * 11000.0f - f[17] * 15000.0f; /* double_three */
    score += f[18] * 400.0f   - f[19] * 450.0f;   /* threat total */

    score += f[20] * 180.0f   - f[21] * 180.0f;   /* center */
    score += f[22] * 160.0f   - f[23] * 160.0f;   /* local adj */
    score += f[24] * 50.0f    - f[25] * 40.0f;    /* pressure */
    score += (f[30] - f[31]) * 300.0f;            /* max span */
    score += 120.0f;                              /* tempo bonus for side_to_move */

    return clamp_int((int)score, -EVAL_MAX_STATIC_SCORE, EVAL_MAX_STATIC_SCORE);
}

int eval_position(const Board *board, Cell side_to_move) {
    EvalFeatures features;
    eval_extract_features(board, side_to_move, &features);

    if (features.values[0] >= 1.0f && features.values[1] < 1.0f) {
        return EVAL_MATE_SCORE - 10;
    }
    if (features.values[1] >= 1.0f && features.values[0] < 1.0f) {
        return -(EVAL_MATE_SCORE - 10);
    }

    return eval_linear_score(&features);
}

static int pattern_score_only(const PatternCounts *c) {
    int s = 0;
    s += c->five * 100000;
    s += c->open_four * 20000;
    s += c->rush_four * 3500;
    s += c->open_three * 2400;
    s += c->sleep_three * 420;
    s += c->open_two * 320;
    s += c->sleep_two * 60;
    return s;
}

int eval_move_heuristic(const Board *board, int row, int col, Cell player) {
    static const int DR[4] = {0, 1, 1, 1};
    static const int DC[4] = {1, 0, 1, -1};

    if (!board_is_empty(board, row, col)) {
        return -1000000;
    }

    Cell opp = board_opponent(player);
    int total = 0;

    for (int d = 0; d < 4; ++d) {
        int my_after[11];
        int opp_after[11];
        my_after[0] = 2;
        opp_after[0] = 2;

        for (int s = -4; s <= 4; ++s) {
            int nr = row + s * DR[d];
            int nc = col + s * DC[d];
            int idx = s + 5;
            if (!board_is_inside(nr, nc)) {
                my_after[idx] = 2;
                opp_after[idx] = 2;
            } else if (s == 0) {
                my_after[idx] = 1;
                opp_after[idx] = 1;
            } else {
                Cell cell = board->cells[nr][nc];
                my_after[idx] = (cell == player) ? 1 : ((cell == opp) ? 2 : 0);
                opp_after[idx] = (cell == opp) ? 1 : ((cell == player) ? 2 : 0);
            }
        }
        my_after[10] = 2;
        opp_after[10] = 2;

        PatternCounts mc = {0};
        PatternCounts oc = {0};
        scan_binary_line(my_after, 11, &mc);
        scan_binary_line(opp_after, 11, &oc);

        total += pattern_score_only(&mc);
        total += (pattern_score_only(&oc) * 9) / 10;
    }

    /* 轻微中心偏好打破同分 */
    int dr = row - 7;
    int dc = col - 7;
    if (dr < 0) dr = -dr;
    if (dc < 0) dc = -dc;
    total += (14 - (dr + dc)) * 8;

    return total;
}
