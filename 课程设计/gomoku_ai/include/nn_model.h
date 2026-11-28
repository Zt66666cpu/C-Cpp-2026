#ifndef GOMOKU_NN_MODEL_H
#define GOMOKU_NN_MODEL_H

#include "board.h"
#include "eval.h"
#include "rules.h"

#include <stdbool.h>
#include <stdint.h>

#define CNN_INPUT_CHANNELS 6

struct NnueModel {
    uint32_t in_dim;
    uint32_t h1_dim;
    uint32_t h2_dim;
    uint32_t out_dim;
    float *in_mean;
    float *in_std;
    float *w1;
    float *b1;
    float *w2;
    float *b2;
    float *w3;
    float *b3;
};

typedef struct {
    float *w1;
    float *b1;
    float *w2;
    float *b2;
} CnnResBlock;

struct CnnModel {
    uint32_t in_ch;
    uint32_t ch;
    uint32_t num_blocks;
    uint32_t pol_mid;
    uint32_t val_mid;
    uint32_t val_fc;

    float *stem_w;
    float *stem_b;
    CnnResBlock *blocks;

    float *pol1_w;
    float *pol1_b;
    float *pol2_w;
    float *pol2_b;

    float *val1_w;
    float *val1_b;
    float *val_fc1_w;
    float *val_fc1_b;
    float *val_fc2_w;
    float *val_fc2_b;
};

typedef struct NnueModel NnueModel;
typedef struct CnnModel CnnModel;

/* 加载与释放 NNUE-Lite 模型 (GNN1 二进制格式) */
NnueModel *nnue_load(const char *path);
void nnue_free(NnueModel *model);
float nnue_forward(const NnueModel *model, const EvalFeatures *features);

/* 加载与释放 Policy-Value CNN 模型 (GCV1 二进制格式) */
CnnModel *cnn_load(const char *path);
void cnn_free(CnnModel *model);

/* 构造 15x15x6 HWC 输入张量（训练采样与 C 推理共用同一逻辑） */
void nn_build_cnn_input(
    Board *board,
    Cell side_to_move,
    Move last_move,
    float *out_hwc
);

void nn_build_cnn_input_ex(
    Board *board,
    Cell side_to_move,
    Move last_move,
    GameRule rule,
    float *out_hwc
);

/* 前向推理：输出 225 维策略 logits（可为 NULL）与胜率估值 [-1, 1] */
float cnn_forward(
    const CnnModel *model,
    const float *input_hwc,
    float *out_policy_logits
);

/* 对合法着法做 Masked Softmax，将策略 logits 转为概率分布 */
void cnn_masked_softmax(
    const float *logits,
    const bool *legal_mask,
    float *out_probs
);

#endif
