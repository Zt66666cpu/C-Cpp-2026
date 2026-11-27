#ifndef GOMOKU_PNNUE_H
#define GOMOKU_PNNUE_H

#include "board.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPN1_MAGIC 0x314E5047U /* "GPN1" little-endian */
#define GPN2_MAGIC 0x324E5047U /* "GPN2" little-endian */
#define PNNUE_DIRS 4
#define PNNUE_NUM_CANON 22143U
#define PNNUE_NUM_RAW_PATTERNS 43923U
#define PNNUE_MAX_DIM 64
#define PNNUE_MAX_V_HIDDEN 64
#define PNNUE_MAX_V2_H1 64
#define PNNUE_MAX_V2_H2 64
#define PNNUE_SUM_SCALE (1.0f / 32.0f)
#define PNNUE_OCC_SCALE (1.0f / 16.0f)

typedef struct PatternNnueModel {
    uint32_t version;
    uint32_t num_canon;
    uint32_t dim;
    uint32_t v_hidden;
    uint32_t v2_h1;
    uint32_t v2_h2;
    float *emb;                                      /* [num_canon * dim] */
    float cell_bias[PNNUE_MAX_DIM];                  /* [dim] */
    float policy_w[2][PNNUE_MAX_DIM];                /* [0=BLACK, 1=WHITE][dim] */
    float policy_b[2];                               /* [0=BLACK, 1=WHITE] */
    float val_w1[PNNUE_MAX_V_HIDDEN][PNNUE_MAX_DIM]; /* GPN1 [v_hidden][dim] */
    float val_b1[2][PNNUE_MAX_V_HIDDEN];             /* GPN1 [0=BLACK, 1=WHITE][v_hidden] */
    float val_w2[PNNUE_MAX_V_HIDDEN];                /* GPN1 [v_hidden] */
    float val_b2[2];                                 /* GPN1 [0=BLACK, 1=WHITE] */
    float v2_w1[PNNUE_MAX_V2_H1][3 * PNNUE_MAX_DIM]; /* GPN2 [v2_h1][3 * dim] */
    float v2_b1[2][PNNUE_MAX_V2_H1];                 /* GPN2 [0=BLACK, 1=WHITE][v2_h1] */
    float v2_w2[PNNUE_MAX_V2_H2][PNNUE_MAX_V2_H1];   /* GPN2 [v2_h2][v2_h1] */
    float v2_b2[2][PNNUE_MAX_V2_H2];                 /* GPN2 [0=BLACK, 1=WHITE][v2_h2] */
    float v2_w3[PNNUE_MAX_V2_H2];                    /* GPN2 [v2_h2] */
    float v2_b3[2];                                  /* GPN2 [0=BLACK, 1=WHITE] */
} PatternNnueModel;

typedef struct PnnueAcc {
    uint8_t cells[BOARD_CELLS];
    uint8_t pool_bucket[BOARD_CELLS];                /* 0=inactive, 1=active_empty, 2=black, 3=white */
    uint16_t pat_id[BOARD_CELLS][PNNUE_DIRS];
    float cell_sum[BOARD_CELLS][PNNUE_MAX_DIM];
    float cell_feat[BOARD_CELLS][PNNUE_MAX_DIM];
    float board_feat_sum[PNNUE_MAX_DIM];
    float pool_sum[4][PNNUE_MAX_DIM];
} PnnueAcc;

void pnnue_init_tables(void);
uint16_t pnnue_encode_cell_dir(const Board *board, int row, int col, int dir);

PatternNnueModel *pnnue_load(const char *path);
void pnnue_free(PatternNnueModel *model);

void pnnue_acc_init(PnnueAcc *acc, const PatternNnueModel *model, const Board *board);
void pnnue_acc_update_cell(
    PnnueAcc *acc,
    const PatternNnueModel *model,
    int row,
    int col,
    Cell new_val
);

float pnnue_eval_value(
    const PnnueAcc *acc,
    const PatternNnueModel *model,
    Cell player
);

float pnnue_eval_policy_cell(
    const PnnueAcc *acc,
    const PatternNnueModel *model,
    int row,
    int col,
    Cell player
);

void pnnue_eval_policy_all(
    const PnnueAcc *acc,
    const PatternNnueModel *model,
    Cell player,
    float out_logits[BOARD_CELLS]
);

#ifdef __cplusplus
}
#endif

#endif
