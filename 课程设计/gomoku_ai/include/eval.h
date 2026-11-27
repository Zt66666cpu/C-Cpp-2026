#ifndef GOMOKU_EVAL_H
#define GOMOKU_EVAL_H

#include "board.h"

#define EVAL_FEATURE_DIM 32
#define EVAL_MATE_SCORE 1000000
#define EVAL_MAX_STATIC_SCORE 80000

typedef struct {
    float values[EVAL_FEATURE_DIM];
} EvalFeatures;

/* 从当前行棋方视角提取 32 维归一化棋型与局势特征 */
void eval_extract_features(
    const Board *board,
    Cell side_to_move,
    EvalFeatures *out
);

/* 使用可解释的线性权重计算静态局面分，范围 [-EVAL_MAX_STATIC_SCORE, EVAL_MAX_STATIC_SCORE] */
int eval_linear_score(const EvalFeatures *features);

/* 对当前局面估值（正值表示对 side_to_move 有利） */
int eval_position(const Board *board, Cell side_to_move);

/* 单点局部启发式打分（用于着法排序，不修改 board） */
int eval_move_heuristic(const Board *board, int row, int col, Cell player);

#endif
