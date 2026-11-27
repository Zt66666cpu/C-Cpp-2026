#ifndef GOMOKU_SELFPLAY_API_H
#define GOMOKU_SELFPLAY_API_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int a_wins;
    int b_wins;
    int draws;
    int illegal_moves;
    double a_avg_time_sec;
    double b_avg_time_sec;
} MatchStats;

/* 预加载模型句柄，供 Python 多进程批量自博弈/评测复用 */
void *gomoku_load_models(const char *nnue_path, const char *cnn_path);
void gomoku_free_models(void *handle);

/* 纯 C 生成一局自博弈训练样本，返回实际记录的步数（<= max_steps），失败返回 -1 */
int gomoku_generate_game(
    void *models_handle,
    int max_depth,
    int vcf_max_depth,
    double time_limit_sec,
    int random_opening_plies,
    float explore_temp,
    int explore_plies,
    float cnn_value_weight,
    uint64_t seed,
    int max_steps,
    uint8_t *out_boards,
    uint8_t *out_players,
    int32_t *out_last_moves,
    uint8_t *out_forbidden,
    float *out_features,
    float *out_policies,
    float *out_search_values,
    float *out_outcomes
);

/* 双边换先公平对局评测：成对开局，交换黑白，返回 0 表示成功 */
int gomoku_run_match(
    void *models_a,
    int a_max_depth,
    int a_vcf_depth,
    double a_time_limit,
    float a_cnn_weight,
    void *models_b,
    int b_max_depth,
    int b_vcf_depth,
    double b_time_limit,
    float b_cnn_weight,
    int num_games,
    uint64_t seed,
    MatchStats *out_stats
);

/* 数值一致性调试接口：对给定棋盘输出 C11 计算的 CNN 输入平面、32 维特征、NNUE/CNN 输出 */
int gomoku_eval_nn_debug(
    const uint8_t *board_cells,
    int player,
    int last_move_idx,
    const char *nnue_path,
    const char *cnn_path,
    float *out_cnn_input_hwc,
    float *out_features_32,
    float *out_nnue_val,
    float *out_cnn_val,
    float *out_cnn_logits
);

/* 提取给定棋盘和执子方的禁手掩码与 32 维 NNUE 特征，返回 0 表示成功 */
int gomoku_extract_state(
    const uint8_t *board_cells,
    int player,
    uint8_t *out_forbidden,
    float *out_features
);

/* 对给定棋盘执行一次搜索，输出最佳落点索引、归一化估值、策略分布、禁手掩码与特征 */
int gomoku_step_search(
    void *models_handle,
    const uint8_t *board_cells,
    int player,
    int last_move_idx,
    int max_depth,
    int vcf_max_depth,
    double time_limit_sec,
    float explore_temp,
    float cnn_value_weight,
    uint64_t seed,
    int *out_best_move,
    float *out_search_value,
    float *out_root_policy,
    uint8_t *out_forbidden,
    float *out_features
);

/* V3 单网 Pattern-NNUE 接口 */
void *gomoku_load_pnnue(const char *pnnue_path);
void gomoku_free_pnnue(void *handle);

int gomoku_eval_pnnue_debug(
    const uint8_t *board_cells,
    int player,
    const char *pnnue_path,
    float *out_value,
    float *out_policy_logits
);

int gomoku_generate_game_pnnue(
    void *pnnue_handle,
    int max_depth,
    int vcf_max_depth,
    double time_limit_sec,
    int random_opening_plies,
    float explore_temp,
    uint64_t seed,
    uint8_t *out_boards,
    uint8_t *out_players,
    uint8_t *out_forbidden,
    float *out_policies,
    float *out_values,
    float *out_outcomes
);

int gomoku_run_match_pnnue(
    void *pnnue_a,
    int a_max_depth,
    int a_vcf_depth,
    double a_time_limit,
    void *pnnue_b,
    void *v2_models_b,
    int b_max_depth,
    int b_vcf_depth,
    double b_time_limit,
    float b_cnn_weight,
    int num_games,
    uint64_t seed,
    MatchStats *out_stats
);

#ifdef __cplusplus
}
#endif

#endif
