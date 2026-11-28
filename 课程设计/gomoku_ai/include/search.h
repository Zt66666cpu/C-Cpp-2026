#ifndef GOMOKU_SEARCH_H
#define GOMOKU_SEARCH_H

#include "board.h"
#include "rules.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct NnueModel NnueModel;
typedef struct CnnModel CnnModel;
typedef struct PatternNnueModel PatternNnueModel;

typedef struct {
    double time_limit_sec;
    int max_depth;
    int vcf_max_depth;
    bool use_nnue;
    bool use_cnn;
    float cnn_value_weight;
    float temperature;
    uint64_t seed;
    GameRule rule;
} SearchConfig;

typedef struct {
    int completed_depth;
    uint64_t nodes;
    uint64_t vcf_nodes;
    double elapsed_sec;
    int score;
    bool used_vcf;
    bool used_guardrail;
    bool used_nnue;
    bool used_cnn;
} SearchStats;

typedef struct {
    Move best_move;
    SearchStats stats;
    bool valid;
    float root_policy[BOARD_CELLS];
} SearchResult;

SearchConfig search_default_config(void);

/* 在给定局面下为 player 寻找最佳合法着法；保证有合法着法时绝不返回非法/禁手位置 */
SearchResult search_find_best_move(
    Board *board,
    Cell player,
    Move last_move,
    const SearchConfig *config,
    const NnueModel *nnue,
    const CnnModel *cnn
);

/* V3 单网 Pattern-NNUE + 持久置换表 + LMR 搜索入口 */
SearchResult search_find_best_move_pnnue(
    Board *board,
    Cell player,
    Move last_move,
    const SearchConfig *config,
    const PatternNnueModel *pnnue
);

/* 收集与已有棋子邻域距离 <= radius 的全部合法候选点（若空盘则仅返回天元） */
int search_generate_candidates(
    Board *board,
    Cell player,
    int radius,
    Move *out_moves,
    int max_moves
);

int search_generate_candidates_ex(
    Board *board,
    Cell player,
    int radius,
    GameRule rule,
    Move *out_moves,
    int max_moves
);

/* VCF（连续冲四必杀）搜索：若存在必胜连冲四路径，返回 true 并输出首步 */
bool search_find_vcf(
    Board *board,
    Cell attacker,
    int max_depth,
    double time_limit_sec,
    Move *win_move,
    uint64_t *out_nodes
);

bool search_find_vcf_ex(
    Board *board,
    Cell attacker,
    int max_depth,
    double time_limit_sec,
    GameRule rule,
    Move *win_move,
    uint64_t *out_nodes
);

/* 计算当前局面的 64 位 Zobrist 哈希（供置换表与测试验证） */
uint64_t search_compute_zobrist(const Board *board, Cell side_to_move);

#endif
