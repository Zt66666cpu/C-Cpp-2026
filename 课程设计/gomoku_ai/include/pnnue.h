#ifndef GOMOKU_PNNUE_H
#define GOMOKU_PNNUE_H

#include "board.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPN1_MAGIC 0x314E5047U /* "GPN1" little-endian */
#define PNNUE_DIRS 4
#define PNNUE_NUM_CANON 22143U
#define PNNUE_NUM_RAW_PATTERNS 43923U
#define PNNUE_MAX_DIM 64
#define PNNUE_MAX_V_HIDDEN 64
#define PNNUE_SUM_SCALE (1.0f / 32.0f)

typedef struct PatternNnueModel {
    uint32_t num_canon;
    uint32_t dim;
    uint32_t v_hidden;
    float *emb;                                      /* [num_canon * dim] */
    float cell_bias[PNNUE_MAX_DIM];                  /* [dim] */
    float policy_w[2][PNNUE_MAX_DIM];                /* [0=BLACK, 1=WHITE][dim] */
    float policy_b[2];                               /* [0=BLACK, 1=WHITE] */
    float val_w1[PNNUE_MAX_V_HIDDEN][PNNUE_MAX_DIM]; /* [v_hidden][dim] */
    float val_b1[2][PNNUE_MAX_V_HIDDEN];             /* [0=BLACK, 1=WHITE][v_hidden] */
    float val_w2[PNNUE_MAX_V_HIDDEN];                /* [v_hidden] */
    float val_b2[2];                                 /* [0=BLACK, 1=WHITE] */
} PatternNnueModel;

typedef struct PnnueAcc {
    uint8_t cells[BOARD_CELLS];
    uint16_t pat_id[BOARD_CELLS][PNNUE_DIRS];
    float cell_sum[BOARD_CELLS][PNNUE_MAX_DIM];
    float cell_feat[BOARD_CELLS][PNNUE_MAX_DIM];
    float board_feat_sum[PNNUE_MAX_DIM];
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
