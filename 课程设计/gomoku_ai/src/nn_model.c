#include "nn_model.h"

#include "rules.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GNN1_MAGIC 0x314E4E47U /* "GNN1" */
#define GCV1_MAGIC 0x31564347U /* "GCV1" */

static bool read_u32_le(FILE *fp, uint32_t *out) {
    uint8_t b[4];
    if (fread(b, 1, 4, fp) != 4) {
        return false;
    }
    *out = (uint32_t)b[0]
        | ((uint32_t)b[1] << 8)
        | ((uint32_t)b[2] << 16)
        | ((uint32_t)b[3] << 24);
    return true;
}

static float *read_float_array(FILE *fp, size_t count) {
    if (count == 0) {
        return NULL;
    }
    float *arr = malloc(count * sizeof(float));
    if (!arr) {
        return NULL;
    }
    if (fread(arr, sizeof(float), count, fp) != count) {
        free(arr);
        return NULL;
    }
    return arr;
}

static float relu_f(float x) {
    return x > 0.0f ? x : 0.0f;
}

/* ==================== NNUE-Lite MLP ==================== */

NnueModel *nnue_load(const char *path) {
    if (!path) {
        return NULL;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return NULL;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t in_dim = 0;
    uint32_t h1_dim = 0;
    uint32_t h2_dim = 0;
    uint32_t out_dim = 0;

    if (!read_u32_le(fp, &magic)
        || !read_u32_le(fp, &version)
        || !read_u32_le(fp, &in_dim)
        || !read_u32_le(fp, &h1_dim)
        || !read_u32_le(fp, &h2_dim)
        || !read_u32_le(fp, &out_dim)) {
        fclose(fp);
        return NULL;
    }

    if (magic != GNN1_MAGIC || version != 1 || in_dim != EVAL_FEATURE_DIM
        || out_dim != 1 || h1_dim == 0 || h2_dim == 0
        || h1_dim > 1024 || h2_dim > 1024) {
        fclose(fp);
        return NULL;
    }

    NnueModel *m = calloc(1, sizeof(NnueModel));
    if (!m) {
        fclose(fp);
        return NULL;
    }
    m->in_dim = in_dim;
    m->h1_dim = h1_dim;
    m->h2_dim = h2_dim;
    m->out_dim = out_dim;

    m->in_mean = read_float_array(fp, in_dim);
    m->in_std = read_float_array(fp, in_dim);
    m->w1 = read_float_array(fp, (size_t)h1_dim * in_dim);
    m->b1 = read_float_array(fp, h1_dim);
    m->w2 = read_float_array(fp, (size_t)h2_dim * h1_dim);
    m->b2 = read_float_array(fp, h2_dim);
    m->w3 = read_float_array(fp, (size_t)out_dim * h2_dim);
    m->b3 = read_float_array(fp, out_dim);
    fclose(fp);

    if (!m->in_mean || !m->in_std || !m->w1 || !m->b1
        || !m->w2 || !m->b2 || !m->w3 || !m->b3) {
        nnue_free(m);
        return NULL;
    }

    return m;
}

void nnue_free(NnueModel *model) {
    if (!model) {
        return;
    }
    free(model->in_mean);
    free(model->in_std);
    free(model->w1);
    free(model->b1);
    free(model->w2);
    free(model->b2);
    free(model->w3);
    free(model->b3);
    free(model);
}

float nnue_forward(const NnueModel *model, const EvalFeatures *features) {
    if (!model || !features) {
        return 0.0f;
    }

    float z[EVAL_FEATURE_DIM];
    for (uint32_t i = 0; i < model->in_dim; ++i) {
        float std = model->in_std[i];
        if (std < 1e-5f) std = 1.0f;
        z[i] = (features->values[i] - model->in_mean[i]) / std;
    }

    float h1[1024];
    for (uint32_t j = 0; j < model->h1_dim; ++j) {
        const float *row = model->w1 + (size_t)j * model->in_dim;
        float acc = model->b1[j];
        for (uint32_t i = 0; i < model->in_dim; ++i) {
            acc += row[i] * z[i];
        }
        h1[j] = relu_f(acc);
    }

    float h2[1024];
    for (uint32_t k = 0; k < model->h2_dim; ++k) {
        const float *row = model->w2 + (size_t)k * model->h1_dim;
        float acc = model->b2[k];
        for (uint32_t j = 0; j < model->h1_dim; ++j) {
            acc += row[j] * h1[j];
        }
        h2[k] = relu_f(acc);
    }

    float out = model->b3[0];
    for (uint32_t k = 0; k < model->h2_dim; ++k) {
        out += model->w3[k] * h2[k];
    }
    return tanhf(out);
}

/* ==================== Policy-Value CNN ==================== */

CnnModel *cnn_load(const char *path) {
    if (!path) {
        return NULL;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        return NULL;
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    uint32_t in_ch = 0;
    uint32_t ch = 0;
    uint32_t num_blocks = 0;
    uint32_t pol_mid = 0;
    uint32_t val_mid = 0;
    uint32_t val_fc = 0;

    if (!read_u32_le(fp, &magic)
        || !read_u32_le(fp, &version)
        || !read_u32_le(fp, &in_ch)
        || !read_u32_le(fp, &ch)
        || !read_u32_le(fp, &num_blocks)
        || !read_u32_le(fp, &pol_mid)
        || !read_u32_le(fp, &val_mid)
        || !read_u32_le(fp, &val_fc)) {
        fclose(fp);
        return NULL;
    }

    if (magic != GCV1_MAGIC || version != 1 || in_ch != CNN_INPUT_CHANNELS
        || ch == 0 || ch > 256 || num_blocks == 0 || num_blocks > 20
        || pol_mid == 0 || val_mid == 0 || val_fc == 0) {
        fclose(fp);
        return NULL;
    }

    CnnModel *m = calloc(1, sizeof(CnnModel));
    if (!m) {
        fclose(fp);
        return NULL;
    }
    m->in_ch = in_ch;
    m->ch = ch;
    m->num_blocks = num_blocks;
    m->pol_mid = pol_mid;
    m->val_mid = val_mid;
    m->val_fc = val_fc;

    m->stem_w = read_float_array(fp, (size_t)ch * 9U * in_ch);
    m->stem_b = read_float_array(fp, ch);
    m->blocks = calloc(num_blocks, sizeof(CnnResBlock));
    if (!m->stem_w || !m->stem_b || !m->blocks) {
        fclose(fp);
        cnn_free(m);
        return NULL;
    }

    for (uint32_t b = 0; b < num_blocks; ++b) {
        m->blocks[b].w1 = read_float_array(fp, (size_t)ch * 9U * ch);
        m->blocks[b].b1 = read_float_array(fp, ch);
        m->blocks[b].w2 = read_float_array(fp, (size_t)ch * 9U * ch);
        m->blocks[b].b2 = read_float_array(fp, ch);
        if (!m->blocks[b].w1 || !m->blocks[b].b1
            || !m->blocks[b].w2 || !m->blocks[b].b2) {
            fclose(fp);
            cnn_free(m);
            return NULL;
        }
    }

    m->pol1_w = read_float_array(fp, (size_t)pol_mid * ch);
    m->pol1_b = read_float_array(fp, pol_mid);
    m->pol2_w = read_float_array(fp, (size_t)pol_mid);
    m->pol2_b = read_float_array(fp, 1);

    m->val1_w = read_float_array(fp, (size_t)val_mid * ch);
    m->val1_b = read_float_array(fp, val_mid);
    m->val_fc1_w = read_float_array(fp, (size_t)val_fc * val_mid * BOARD_CELLS);
    m->val_fc1_b = read_float_array(fp, val_fc);
    m->val_fc2_w = read_float_array(fp, val_fc);
    m->val_fc2_b = read_float_array(fp, 1);
    fclose(fp);

    if (!m->pol1_w || !m->pol1_b || !m->pol2_w || !m->pol2_b
        || !m->val1_w || !m->val1_b || !m->val_fc1_w || !m->val_fc1_b
        || !m->val_fc2_w || !m->val_fc2_b) {
        cnn_free(m);
        return NULL;
    }

    return m;
}

void cnn_free(CnnModel *model) {
    if (!model) {
        return;
    }
    free(model->stem_w);
    free(model->stem_b);
    if (model->blocks) {
        for (uint32_t b = 0; b < model->num_blocks; ++b) {
            free(model->blocks[b].w1);
            free(model->blocks[b].b1);
            free(model->blocks[b].w2);
            free(model->blocks[b].b2);
        }
        free(model->blocks);
    }
    free(model->pol1_w);
    free(model->pol1_b);
    free(model->pol2_w);
    free(model->pol2_b);
    free(model->val1_w);
    free(model->val1_b);
    free(model->val_fc1_w);
    free(model->val_fc1_b);
    free(model->val_fc2_w);
    free(model->val_fc2_b);
    free(model);
}

void nn_build_cnn_input(
    Board *board,
    Cell side_to_move,
    Move last_move,
    float *out_hwc
) {
    Cell my = (side_to_move == CELL_BLACK || side_to_move == CELL_WHITE)
        ? side_to_move
        : CELL_BLACK;
    Cell opp = board_opponent(my);

    int black_count = 0;
    if (my == CELL_BLACK) {
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                if (board->cells[r][c] == CELL_BLACK) {
                    ++black_count;
                }
            }
        }
    }

    bool check_forbidden = (my == CELL_BLACK && black_count >= 4);
    float is_black_plane = (my == CELL_BLACK) ? 1.0f : 0.0f;

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            Cell cell = board->cells[r][c];
            float *p = out_hwc + ((size_t)r * BOARD_SIZE + (size_t)c) * CNN_INPUT_CHANNELS;
            p[0] = (cell == my) ? 1.0f : 0.0f;
            p[1] = (cell == opp) ? 1.0f : 0.0f;
            p[2] = (cell == CELL_EMPTY) ? 1.0f : 0.0f;
            p[3] = (r == last_move.row && c == last_move.col) ? 1.0f : 0.0f;
            p[4] = is_black_plane;
            p[5] = (check_forbidden
                    && cell == CELL_EMPTY
                    && rules_check_black_forbidden(board, r, c) != FORBIDDEN_NONE)
                ? 1.0f
                : 0.0f;
        }
    }
}

static void im2col_3x3_hwc(
    const float * restrict in,
    uint32_t cin,
    float * restrict col
) {
    size_t row_stride = (size_t)cin * 9U;
    size_t cin_bytes = (size_t)cin * sizeof(float);

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            float * restrict dst_row = col + ((size_t)r * BOARD_SIZE + (size_t)c) * row_stride;
            for (int kr = 0; kr < 3; ++kr) {
                int nr = r + kr - 1;
                if (nr < 0 || nr >= BOARD_SIZE) {
                    memset(dst_row + (size_t)kr * 3U * cin, 0, 3U * cin_bytes);
                    continue;
                }
                for (int kc = 0; kc < 3; ++kc) {
                    int nc = c + kc - 1;
                    float * restrict dst = dst_row + (size_t)(kr * 3 + kc) * cin;
                    if (nc < 0 || nc >= BOARD_SIZE) {
                        memset(dst, 0, cin_bytes);
                    } else {
                        const float * restrict src =
                            in + ((size_t)nr * BOARD_SIZE + (size_t)nc) * cin;
                        memcpy(dst, src, cin_bytes);
                    }
                }
            }
        }
    }
}

/* 4x4 寄存器分块矩阵乘：out[M][N] = A[M][K] * W[N][K]^T + b[N] (+ residual)，可选 ReLU */
static void gemm_nt_4x4(
    const float * restrict A,
    uint32_t M,
    uint32_t K,
    const float * restrict W,
    const float * restrict b,
    uint32_t N,
    const float * restrict residual,
    bool apply_relu,
    float * restrict out
) {
    uint32_t s = 0;
    for (; s + 4U <= M; s += 4U) {
        const float * restrict a0 = A + (size_t)(s + 0U) * K;
        const float * restrict a1 = A + (size_t)(s + 1U) * K;
        const float * restrict a2 = A + (size_t)(s + 2U) * K;
        const float * restrict a3 = A + (size_t)(s + 3U) * K;

        uint32_t co = 0;
        for (; co + 4U <= N; co += 4U) {
            const float * restrict w0 = W + (size_t)(co + 0U) * K;
            const float * restrict w1 = W + (size_t)(co + 1U) * K;
            const float * restrict w2 = W + (size_t)(co + 2U) * K;
            const float * restrict w3 = W + (size_t)(co + 3U) * K;

            float c00 = b[co + 0U], c01 = b[co + 1U], c02 = b[co + 2U], c03 = b[co + 3U];
            float c10 = b[co + 0U], c11 = b[co + 1U], c12 = b[co + 2U], c13 = b[co + 3U];
            float c20 = b[co + 0U], c21 = b[co + 1U], c22 = b[co + 2U], c23 = b[co + 3U];
            float c30 = b[co + 0U], c31 = b[co + 1U], c32 = b[co + 2U], c33 = b[co + 3U];

            for (uint32_t k = 0; k < K; ++k) {
                float va0 = a0[k], va1 = a1[k], va2 = a2[k], va3 = a3[k];
                float vw0 = w0[k], vw1 = w1[k], vw2 = w2[k], vw3 = w3[k];
                c00 += va0 * vw0; c01 += va0 * vw1; c02 += va0 * vw2; c03 += va0 * vw3;
                c10 += va1 * vw0; c11 += va1 * vw1; c12 += va1 * vw2; c13 += va1 * vw3;
                c20 += va2 * vw0; c21 += va2 * vw1; c22 += va2 * vw2; c23 += va2 * vw3;
                c30 += va3 * vw0; c31 += va3 * vw1; c32 += va3 * vw2; c33 += va3 * vw3;
            }

            float * restrict d0 = out + (size_t)(s + 0U) * N + co;
            float * restrict d1 = out + (size_t)(s + 1U) * N + co;
            float * restrict d2 = out + (size_t)(s + 2U) * N + co;
            float * restrict d3 = out + (size_t)(s + 3U) * N + co;

            if (residual) {
                const float * restrict r0 = residual + (size_t)(s + 0U) * N + co;
                const float * restrict r1 = residual + (size_t)(s + 1U) * N + co;
                const float * restrict r2 = residual + (size_t)(s + 2U) * N + co;
                const float * restrict r3 = residual + (size_t)(s + 3U) * N + co;
                c00 += r0[0]; c01 += r0[1]; c02 += r0[2]; c03 += r0[3];
                c10 += r1[0]; c11 += r1[1]; c12 += r1[2]; c13 += r1[3];
                c20 += r2[0]; c21 += r2[1]; c22 += r2[2]; c23 += r2[3];
                c30 += r3[0]; c31 += r3[1]; c32 += r3[2]; c33 += r3[3];
            }

            if (apply_relu) {
                c00 = relu_f(c00); c01 = relu_f(c01); c02 = relu_f(c02); c03 = relu_f(c03);
                c10 = relu_f(c10); c11 = relu_f(c11); c12 = relu_f(c12); c13 = relu_f(c13);
                c20 = relu_f(c20); c21 = relu_f(c21); c22 = relu_f(c22); c23 = relu_f(c23);
                c30 = relu_f(c30); c31 = relu_f(c31); c32 = relu_f(c32); c33 = relu_f(c33);
            }

            d0[0] = c00; d0[1] = c01; d0[2] = c02; d0[3] = c03;
            d1[0] = c10; d1[1] = c11; d1[2] = c12; d1[3] = c13;
            d2[0] = c20; d2[1] = c21; d2[2] = c22; d2[3] = c23;
            d3[0] = c30; d3[1] = c31; d3[2] = c32; d3[3] = c33;
        }

        for (; co < N; ++co) {
            const float * restrict wc = W + (size_t)co * K;
            float c0 = b[co], c1 = b[co], c2 = b[co], c3 = b[co];
            for (uint32_t k = 0; k < K; ++k) {
                float wk = wc[k];
                c0 += a0[k] * wk;
                c1 += a1[k] * wk;
                c2 += a2[k] * wk;
                c3 += a3[k] * wk;
            }
            if (residual) {
                c0 += residual[(size_t)(s + 0U) * N + co];
                c1 += residual[(size_t)(s + 1U) * N + co];
                c2 += residual[(size_t)(s + 2U) * N + co];
                c3 += residual[(size_t)(s + 3U) * N + co];
            }
            if (apply_relu) {
                c0 = relu_f(c0); c1 = relu_f(c1); c2 = relu_f(c2); c3 = relu_f(c3);
            }
            out[(size_t)(s + 0U) * N + co] = c0;
            out[(size_t)(s + 1U) * N + co] = c1;
            out[(size_t)(s + 2U) * N + co] = c2;
            out[(size_t)(s + 3U) * N + co] = c3;
        }
    }

    for (; s < M; ++s) {
        const float * restrict as = A + (size_t)s * K;
        float * restrict ds = out + (size_t)s * N;
        const float * restrict rs = residual ? (residual + (size_t)s * N) : NULL;
        for (uint32_t co = 0; co < N; ++co) {
            const float * restrict wc = W + (size_t)co * K;
            float acc = b[co];
            for (uint32_t k = 0; k < K; ++k) {
                acc += as[k] * wc[k];
            }
            if (rs) acc += rs[co];
            ds[co] = apply_relu ? relu_f(acc) : acc;
        }
    }
}

static void conv3x3_hwc(
    const float * restrict in,
    uint32_t cin,
    const float * restrict w,
    const float * restrict b,
    uint32_t cout,
    const float * restrict residual,
    bool apply_relu,
    float * restrict col_scratch,
    float * restrict out
) {
    im2col_3x3_hwc(in, cin, col_scratch);
    gemm_nt_4x4(
        col_scratch, BOARD_CELLS, cin * 9U, w, b, cout, residual, apply_relu, out
    );
}

static void conv1x1_hwc(
    const float * restrict in,
    uint32_t cin,
    const float * restrict w,
    const float * restrict b,
    uint32_t cout,
    bool apply_relu,
    float * restrict out
) {
    gemm_nt_4x4(in, BOARD_CELLS, cin, w, b, cout, NULL, apply_relu, out);
}

float cnn_forward(
    const CnnModel *model,
    const float *input_hwc,
    float *out_policy_logits
) {
    if (!model || !input_hwc) {
        return 0.0f;
    }

    uint32_t ch = model->ch;
    uint32_t max_in = model->in_ch > ch ? model->in_ch : ch;
    uint32_t max_head = model->pol_mid > model->val_mid ? model->pol_mid : model->val_mid;
    size_t buf_plane = (size_t)BOARD_CELLS * ch;
    size_t col_plane = (size_t)BOARD_CELLS * 9U * max_in;
    size_t head_plane = (size_t)BOARD_CELLS * max_head;
    size_t total_floats = buf_plane * 3U + col_plane + head_plane + model->val_fc;

    float *scratch = malloc(total_floats * sizeof(float));
    if (!scratch) {
        return 0.0f;
    }

    float *cur = scratch;
    float *tmp1 = cur + buf_plane;
    float *tmp2 = tmp1 + buf_plane;
    float *col_buf = tmp2 + buf_plane;
    float *head_buf = col_buf + col_plane;
    float *fc_buf = head_buf + head_plane;

    /* Stem */
    conv3x3_hwc(
        input_hwc, model->in_ch, model->stem_w, model->stem_b,
        ch, NULL, true, col_buf, cur
    );

    /* Residual tower */
    for (uint32_t b = 0; b < model->num_blocks; ++b) {
        conv3x3_hwc(
            cur, ch, model->blocks[b].w1, model->blocks[b].b1,
            ch, NULL, true, col_buf, tmp1
        );
        conv3x3_hwc(
            tmp1, ch, model->blocks[b].w2, model->blocks[b].b2,
            ch, cur, true, col_buf, tmp2
        );
        memcpy(cur, tmp2, buf_plane * sizeof(float));
    }

    /* Policy head */
    if (out_policy_logits) {
        conv1x1_hwc(
            cur, ch, model->pol1_w, model->pol1_b,
            model->pol_mid, true, head_buf
        );
        conv1x1_hwc(
            head_buf, model->pol_mid, model->pol2_w, model->pol2_b,
            1, false, out_policy_logits
        );
    }

    /* Value head */
    conv1x1_hwc(
        cur, ch, model->val1_w, model->val1_b,
        model->val_mid, true, head_buf
    );

    /* 与 PyTorch NCHW flatten 顺序一致：c * 225 + r * 15 + col */
    uint32_t val_flat_dim = model->val_mid * BOARD_CELLS;
    float *val_chw = tmp1; /* tmp1 至少有 BOARD_CELLS * ch >= val_mid * BOARD_CELLS */
    for (uint32_t r = 0; r < BOARD_SIZE; ++r) {
        for (uint32_t c = 0; c < BOARD_SIZE; ++c) {
            size_t spatial = (size_t)r * BOARD_SIZE + c;
            const float *src = head_buf + spatial * model->val_mid;
            for (uint32_t k = 0; k < model->val_mid; ++k) {
                val_chw[(size_t)k * BOARD_CELLS + spatial] = src[k];
            }
        }
    }

    for (uint32_t o = 0; o < model->val_fc; ++o) {
        const float *row = model->val_fc1_w + (size_t)o * val_flat_dim;
        float acc = model->val_fc1_b[o];
        for (uint32_t i = 0; i < val_flat_dim; ++i) {
            acc += row[i] * val_chw[i];
        }
        fc_buf[o] = relu_f(acc);
    }

    float val_out = model->val_fc2_b[0];
    for (uint32_t o = 0; o < model->val_fc; ++o) {
        val_out += model->val_fc2_w[o] * fc_buf[o];
    }

    free(scratch);
    return tanhf(val_out);
}

void cnn_masked_softmax(
    const float *logits,
    const bool *legal_mask,
    float *out_probs
) {
    float max_logit = -1e30f;
    bool any_legal = false;
    for (int i = 0; i < BOARD_CELLS; ++i) {
        if (legal_mask[i]) {
            if (!any_legal || logits[i] > max_logit) {
                max_logit = logits[i];
                any_legal = true;
            }
        }
    }
    if (!any_legal) {
        memset(out_probs, 0, sizeof(float) * BOARD_CELLS);
        return;
    }

    double sum = 0.0;
    for (int i = 0; i < BOARD_CELLS; ++i) {
        if (legal_mask[i]) {
            double e = exp((double)(logits[i] - max_logit));
            out_probs[i] = (float)e;
            sum += e;
        } else {
            out_probs[i] = 0.0f;
        }
    }
    if (sum > 0.0) {
        for (int i = 0; i < BOARD_CELLS; ++i) {
            out_probs[i] = (float)((double)out_probs[i] / sum);
        }
    }
}
