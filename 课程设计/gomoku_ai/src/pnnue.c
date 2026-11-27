#include "pnnue.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const int kDirDr[PNNUE_DIRS] = {0, 1, 1, 1};
static const int kDirDc[PNNUE_DIRS] = {1, 0, 1, -1};
static const uint32_t kPow3[10] = {
    1U, 3U, 9U, 27U, 81U, 243U, 729U, 2187U, 6561U, 19683U
};

static bool g_tables_ready = false;
static uint32_t g_lr_offset[5][5];
static uint16_t g_raw_to_canon[PNNUE_NUM_RAW_PATTERNS];

static bool read_u32_le(FILE *fp, uint32_t *out) {
    uint8_t buf[4];
    if (fread(buf, 1, 4, fp) != 4) return false;
    *out = (uint32_t)buf[0]
         | ((uint32_t)buf[1] << 8)
         | ((uint32_t)buf[2] << 16)
         | ((uint32_t)buf[3] << 24);
    return true;
}

static bool read_float_array(FILE *fp, float *dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = 0;
        if (!read_u32_le(fp, &bits)) return false;
        memcpy(&dst[i], &bits, sizeof(float));
    }
    return true;
}

static inline float relu_f(float x) {
    return x > 0.0f ? x : 0.0f;
}

void pnnue_init_tables(void) {
    if (g_tables_ready) return;

    uint32_t raw_idx = 0;
    uint32_t canon_idx = 0;

    for (int L = 0; L <= 4; ++L) {
        for (int R = 0; R <= 4; ++R) {
            int len = L + 1 + R;
            uint32_t max_code = kPow3[len];
            g_lr_offset[L][R] = raw_idx;
            for (uint32_t code = 0; code < max_code; ++code) {
                uint32_t tmp = code;
                uint32_t rev = 0;
                for (int t = 0; t < len; ++t) {
                    rev = rev * 3U + (tmp % 3U);
                    tmp /= 3U;
                }
                if (L < R || (L == R && code <= rev)) {
                    g_raw_to_canon[raw_idx] = (uint16_t)canon_idx;
                    ++canon_idx;
                } else {
                    uint32_t mirror_raw = g_lr_offset[R][L] + rev;
                    g_raw_to_canon[raw_idx] = g_raw_to_canon[mirror_raw];
                }
                ++raw_idx;
            }
        }
    }

    g_tables_ready = true;
}

static uint16_t encode_from_cells(const uint8_t cells[BOARD_CELLS], int row, int col, int dir) {
    int dr = kDirDr[dir];
    int dc = kDirDc[dir];

    int L = 0;
    while (L < 4) {
        int rr = row - (L + 1) * dr;
        int cc = col - (L + 1) * dc;
        if (rr < 0 || rr >= BOARD_SIZE || cc < 0 || cc >= BOARD_SIZE) break;
        ++L;
    }

    int R = 0;
    while (R < 4) {
        int rr = row + (R + 1) * dr;
        int cc = col + (R + 1) * dc;
        if (rr < 0 || rr >= BOARD_SIZE || cc < 0 || cc >= BOARD_SIZE) break;
        ++R;
    }

    uint32_t code = 0;
    for (int k = -L; k <= R; ++k) {
        int rr = row + k * dr;
        int cc = col + k * dc;
        code = code * 3U + (uint32_t)cells[rr * BOARD_SIZE + cc];
    }

    return g_raw_to_canon[g_lr_offset[L][R] + code];
}

uint16_t pnnue_encode_cell_dir(const Board *board, int row, int col, int dir) {
    pnnue_init_tables();
    if (!board || row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE ||
        dir < 0 || dir >= PNNUE_DIRS) {
        return 0;
    }
    int dr = kDirDr[dir];
    int dc = kDirDc[dir];

    int L = 0;
    while (L < 4) {
        int rr = row - (L + 1) * dr;
        int cc = col - (L + 1) * dc;
        if (rr < 0 || rr >= BOARD_SIZE || cc < 0 || cc >= BOARD_SIZE) break;
        ++L;
    }

    int R = 0;
    while (R < 4) {
        int rr = row + (R + 1) * dr;
        int cc = col + (R + 1) * dc;
        if (rr < 0 || rr >= BOARD_SIZE || cc < 0 || cc >= BOARD_SIZE) break;
        ++R;
    }

    uint32_t code = 0;
    for (int k = -L; k <= R; ++k) {
        int rr = row + k * dr;
        int cc = col + k * dc;
        code = code * 3U + (uint32_t)board->cells[rr][cc];
    }

    return g_raw_to_canon[g_lr_offset[L][R] + code];
}

PatternNnueModel *pnnue_load(const char *path) {
    if (!path) return NULL;
    pnnue_init_tables();

    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    uint32_t magic = 0;
    uint32_t num_canon = 0;
    uint32_t dim = 0;
    uint32_t v_hidden = 0;
    if (!read_u32_le(fp, &magic) || magic != GPN1_MAGIC ||
        !read_u32_le(fp, &num_canon) ||
        !read_u32_le(fp, &dim) ||
        !read_u32_le(fp, &v_hidden)) {
        fclose(fp);
        return NULL;
    }

    if (num_canon != PNNUE_NUM_CANON ||
        dim == 0 || dim > PNNUE_MAX_DIM ||
        v_hidden == 0 || v_hidden > PNNUE_MAX_V_HIDDEN) {
        fclose(fp);
        return NULL;
    }

    PatternNnueModel *m = (PatternNnueModel *)calloc(1, sizeof(PatternNnueModel));
    if (!m) {
        fclose(fp);
        return NULL;
    }
    m->num_canon = num_canon;
    m->dim = dim;
    m->v_hidden = v_hidden;
    m->emb = (float *)calloc((size_t)num_canon * (size_t)dim, sizeof(float));
    if (!m->emb) {
        free(m);
        fclose(fp);
        return NULL;
    }

    bool ok = read_float_array(fp, m->emb, (size_t)num_canon * (size_t)dim)
           && read_float_array(fp, m->cell_bias, dim)
           && read_float_array(fp, m->policy_w[0], dim)
           && read_float_array(fp, m->policy_w[1], dim)
           && read_float_array(fp, m->policy_b, 2);

    for (uint32_t j = 0; ok && j < v_hidden; ++j) {
        ok = read_float_array(fp, m->val_w1[j], dim);
    }
    if (ok) {
        ok = read_float_array(fp, m->val_b1[0], v_hidden)
          && read_float_array(fp, m->val_b1[1], v_hidden)
          && read_float_array(fp, m->val_w2, v_hidden)
          && read_float_array(fp, m->val_b2, 2);
    }

    fclose(fp);
    if (!ok) {
        pnnue_free(m);
        return NULL;
    }
    return m;
}

void pnnue_free(PatternNnueModel *model) {
    if (!model) return;
    free(model->emb);
    free(model);
}

static void refresh_cell_slot(
    PnnueAcc *acc,
    const PatternNnueModel *model,
    int idx,
    int dir,
    uint16_t new_pat
) {
    uint16_t old_pat = acc->pat_id[idx][dir];
    if (old_pat == new_pat) return;

    uint32_t D = model->dim;
    const float *old_vec = &model->emb[(size_t)old_pat * D];
    const float *new_vec = &model->emb[(size_t)new_pat * D];
    float *csum = acc->cell_sum[idx];
    float *cfeat = acc->cell_feat[idx];
    float *bsum = acc->board_feat_sum;
    const float *cbias = model->cell_bias;

    for (uint32_t k = 0; k < D; ++k) {
        float s = csum[k] + (new_vec[k] - old_vec[k]);
        csum[k] = s;
        float old_f = cfeat[k];
        float new_f = relu_f(s + cbias[k]);
        cfeat[k] = new_f;
        bsum[k] += (new_f - old_f);
    }
    acc->pat_id[idx][dir] = new_pat;
}

void pnnue_acc_init(PnnueAcc *acc, const PatternNnueModel *model, const Board *board) {
    pnnue_init_tables();
    memset(acc, 0, sizeof(*acc));
    if (!model || !board) return;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            acc->cells[r * BOARD_SIZE + c] = (uint8_t)board->cells[r][c];
        }
    }

    uint32_t D = model->dim;
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            int idx = r * BOARD_SIZE + c;
            float *csum = acc->cell_sum[idx];
            float *cfeat = acc->cell_feat[idx];
            for (int d = 0; d < PNNUE_DIRS; ++d) {
                uint16_t pid = encode_from_cells(acc->cells, r, c, d);
                acc->pat_id[idx][d] = pid;
                const float *vec = &model->emb[(size_t)pid * D];
                for (uint32_t k = 0; k < D; ++k) {
                    csum[k] += vec[k];
                }
            }
            for (uint32_t k = 0; k < D; ++k) {
                float f = relu_f(csum[k] + model->cell_bias[k]);
                cfeat[k] = f;
                acc->board_feat_sum[k] += f;
            }
        }
    }
}

void pnnue_acc_update_cell(
    PnnueAcc *acc,
    const PatternNnueModel *model,
    int row,
    int col,
    Cell new_val
) {
    if (!acc || !model) return;
    if (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE) return;

    int center_idx = row * BOARD_SIZE + col;
    uint8_t nv = (uint8_t)new_val;
    if (acc->cells[center_idx] == nv) return;
    acc->cells[center_idx] = nv;

    for (int d = 0; d < PNNUE_DIRS; ++d) {
        int dr = kDirDr[d];
        int dc = kDirDc[d];
        for (int step = -4; step <= 4; ++step) {
            int rr = row + step * dr;
            int cc = col + step * dc;
            if (rr < 0 || rr >= BOARD_SIZE || cc < 0 || cc >= BOARD_SIZE) continue;
            int idx = rr * BOARD_SIZE + cc;
            uint16_t pid = encode_from_cells(acc->cells, rr, cc, d);
            refresh_cell_slot(acc, model, idx, d, pid);
        }
    }
}

float pnnue_eval_value(
    const PnnueAcc *acc,
    const PatternNnueModel *model,
    Cell player
) {
    if (!acc || !model) return 0.0f;
    int p_idx = (player == CELL_BLACK) ? 0 : 1;
    uint32_t D = model->dim;
    uint32_t H = model->v_hidden;

    float v_raw = model->val_b2[p_idx];
    for (uint32_t j = 0; j < H; ++j) {
        const float *w1 = model->val_w1[j];
        float sum = model->val_b1[p_idx][j];
        for (uint32_t k = 0; k < D; ++k) {
            sum += (acc->board_feat_sum[k] * PNNUE_SUM_SCALE) * w1[k];
        }
        v_raw += relu_f(sum) * model->val_w2[j];
    }
    float v = tanhf(v_raw);
    return (p_idx == 0) ? v : -v;
}

float pnnue_eval_policy_cell(
    const PnnueAcc *acc,
    const PatternNnueModel *model,
    int row,
    int col,
    Cell player
) {
    if (!acc || !model) return 0.0f;
    int p_idx = (player == CELL_BLACK) ? 0 : 1;
    uint32_t D = model->dim;
    const float *cfeat = acc->cell_feat[row * BOARD_SIZE + col];
    const float *pw = model->policy_w[p_idx];
    float logit = model->policy_b[p_idx];
    for (uint32_t k = 0; k < D; ++k) {
        logit += cfeat[k] * pw[k];
    }
    return logit;
}

void pnnue_eval_policy_all(
    const PnnueAcc *acc,
    const PatternNnueModel *model,
    Cell player,
    float out_logits[BOARD_CELLS]
) {
    if (!acc || !model || !out_logits) return;
    int p_idx = (player == CELL_BLACK) ? 0 : 1;
    uint32_t D = model->dim;
    const float *pw = model->policy_w[p_idx];
    float pb = model->policy_b[p_idx];

    for (int idx = 0; idx < BOARD_CELLS; ++idx) {
        const float *cfeat = acc->cell_feat[idx];
        float logit = pb;
        for (uint32_t k = 0; k < D; ++k) {
            logit += cfeat[k] * pw[k];
        }
        out_logits[idx] = logit;
    }
}
