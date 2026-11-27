#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#define NOUSER
#include <windows.h>
#else
#include <pthread.h>
#endif

#include "board.h"
#include "eval.h"
#include "nn_model.h"
#include "pnnue.h"
#include "rules.h"
#include "search.h"

#include "raylib.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GUI_WIDTH 1120
#define GUI_HEIGHT 800
#define BOARD_RECT_X 24
#define BOARD_RECT_Y 28
#define BOARD_RECT_SIZE 724
#define GRID_ORIGIN_X 64
#define GRID_ORIGIN_Y 68
#define CELL_PX 46
#define SIDEBAR_X 768
#define SIDEBAR_Y 20
#define SIDEBAR_W 332
#define SIDEBAR_H 760

typedef enum {
    GUI_MODE_HUMAN_BLACK = 0,
    GUI_MODE_HUMAN_WHITE,
    GUI_MODE_TWO_PLAYER,
    GUI_MODE_AI_VS_AI
} GuiGameMode;

typedef enum {
    ENGINE_V3_PNNUE = 0,
    ENGINE_V2_DUAL,
    ENGINE_HEURISTIC
} GuiEngineKind;

typedef struct {
    Move move;
    Cell player;
    bool is_ai;
    int depth;
    double elapsed_sec;
    int score;
    bool used_vcf;
    bool used_guardrail;
} GuiMoveRecord;

typedef struct {
#if defined(_WIN32)
    CRITICAL_SECTION cs;
#else
    pthread_mutex_t mu;
#endif
    bool busy;
    bool has_result;
    uint64_t result_gen;
    bool result_is_hint;
    SearchResult result;
} AiShared;

typedef struct {
    Board board;
    Cell player;
    Move last_move;
    SearchConfig cfg;
    GuiEngineKind engine;
    const PatternNnueModel *pnnue;
    const NnueModel *nnue;
    const CnnModel *cnn;
    uint64_t generation;
    bool is_hint;
    AiShared *shared;
} AiWorkerArgs;

typedef struct {
    Board board;
    GuiMoveRecord history[BOARD_CELLS];
    int history_len;
    Cell current_player;
    Move last_move;
    Move hint_move;
    bool has_hint;

    GuiGameMode mode;
    GuiEngineKind engine;
    int max_depth;
    int time_idx;
    bool show_forbidden;
    bool show_numbers;

    bool game_over;
    MoveOutcome final_outcome;
    char status_banner[128];
    char last_ai_info[128];
    char toast_msg[128];
    double toast_until;
    float black_win_prob;

    bool forbidden_map[BOARD_CELLS];
    ForbiddenReason forbidden_reasons[BOARD_CELLS];

    uint64_t generation;
    double think_start_sec;
    AiShared shared;

    PatternNnueModel *pnnue;
    NnueModel *nnue;
    CnnModel *cnn;
    Font ui_font;
    bool custom_font;
} GuiState;

static const double kTimeOptions[3] = {1.0, 3.0, 5.0};
static const int kDepthOptions[4] = {3, 4, 5, 6};

static void gui_mutex_init(AiShared *s) {
#if defined(_WIN32)
    InitializeCriticalSection(&s->cs);
#else
    pthread_mutex_init(&s->mu, NULL);
#endif
}

static void gui_mutex_destroy(AiShared *s) {
#if defined(_WIN32)
    DeleteCriticalSection(&s->cs);
#else
    pthread_mutex_destroy(&s->mu);
#endif
}

static void gui_mutex_lock(AiShared *s) {
#if defined(_WIN32)
    EnterCriticalSection(&s->cs);
#else
    pthread_mutex_lock(&s->mu);
#endif
}

static void gui_mutex_unlock(AiShared *s) {
#if defined(_WIN32)
    LeaveCriticalSection(&s->cs);
#else
    pthread_mutex_unlock(&s->mu);
#endif
}

#if defined(_WIN32)
static DWORD WINAPI ai_worker_proc(LPVOID raw) {
#else
static void *ai_worker_proc(void *raw) {
#endif
    AiWorkerArgs *args = (AiWorkerArgs *)raw;
    SearchResult res;
    if (args->engine == ENGINE_V3_PNNUE && args->pnnue != NULL) {
        res = search_find_best_move_pnnue(
            &args->board, args->player, args->last_move, &args->cfg, args->pnnue
        );
    } else if (args->engine == ENGINE_V2_DUAL && (args->nnue != NULL || args->cnn != NULL)) {
        res = search_find_best_move(
            &args->board, args->player, args->last_move, &args->cfg, args->nnue, args->cnn
        );
    } else {
        SearchConfig heur_cfg = args->cfg;
        heur_cfg.use_nnue = false;
        heur_cfg.use_cnn = false;
        res = search_find_best_move(
            &args->board, args->player, args->last_move, &heur_cfg, NULL, NULL
        );
    }

    gui_mutex_lock(args->shared);
    args->shared->result = res;
    args->shared->result_gen = args->generation;
    args->shared->result_is_hint = args->is_hint;
    args->shared->has_result = true;
    args->shared->busy = false;
    gui_mutex_unlock(args->shared);

    free(args);
#if defined(_WIN32)
    return 0;
#else
    return NULL;
#endif
}

static bool spawn_ai_worker(GuiState *st, bool is_hint) {
    gui_mutex_lock(&st->shared);
    bool busy = st->shared.busy;
    gui_mutex_unlock(&st->shared);
    if (busy) {
        return false;
    }

    AiWorkerArgs *args = (AiWorkerArgs *)calloc(1, sizeof(AiWorkerArgs));
    if (!args) {
        return false;
    }
    args->board = st->board;
    args->player = st->current_player;
    args->last_move = st->last_move;
    args->cfg = search_default_config();
    args->cfg.max_depth = st->max_depth;
    args->cfg.vcf_max_depth = 14;
    args->cfg.time_limit_sec = kTimeOptions[st->time_idx];
    args->cfg.cnn_value_weight = 0.35f;
    args->cfg.use_nnue = (st->engine != ENGINE_HEURISTIC);
    args->cfg.use_cnn = (st->engine == ENGINE_V2_DUAL && st->cnn != NULL);
    args->engine = st->engine;
    args->pnnue = st->pnnue;
    args->nnue = st->nnue;
    args->cnn = st->cnn;
    args->generation = st->generation;
    args->is_hint = is_hint;
    args->shared = &st->shared;

    gui_mutex_lock(&st->shared);
    st->shared.busy = true;
    st->shared.has_result = false;
    gui_mutex_unlock(&st->shared);
    st->think_start_sec = GetTime();

#if defined(_WIN32)
    HANDLE th = CreateThread(NULL, 0, ai_worker_proc, args, 0, NULL);
    if (!th) {
        gui_mutex_lock(&st->shared);
        st->shared.busy = false;
        gui_mutex_unlock(&st->shared);
        free(args);
        return false;
    }
    CloseHandle(th);
#else
    pthread_t tid;
    if (pthread_create(&tid, NULL, ai_worker_proc, args) != 0) {
        gui_mutex_lock(&st->shared);
        st->shared.busy = false;
        gui_mutex_unlock(&st->shared);
        free(args);
        return false;
    }
    pthread_detach(tid);
#endif
    return true;
}

static bool resolve_model_file(const char *filename, char *out_path, size_t out_sz) {
    const char *app_dir = GetApplicationDirectory();
    char cand[4][1024];
    snprintf(cand[0], sizeof(cand[0]), "models/%s", filename);
    snprintf(cand[1], sizeof(cand[1]), "../models/%s", filename);
    snprintf(cand[2], sizeof(cand[2]), "%smodels/%s", app_dir ? app_dir : "", filename);
    snprintf(cand[3], sizeof(cand[3]), "%s../models/%s", app_dir ? app_dir : "", filename);

    for (int i = 0; i < 4; ++i) {
        if (FileExists(cand[i])) {
            snprintf(out_path, out_sz, "%s", cand[i]);
            return true;
        }
    }
    return false;
}

static void load_best_font(GuiState *st) {
    static const char *kFontPaths[] = {
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    };
    size_t count = sizeof(kFontPaths) / sizeof(kFontPaths[0]);
    for (size_t i = 0; i < count; ++i) {
        if (FileExists(kFontPaths[i])) {
            Font f = LoadFontEx(kFontPaths[i], 28, NULL, 0);
            if (f.texture.id != 0) {
                SetTextureFilter(f.texture, TEXTURE_FILTER_BILINEAR);
                st->ui_font = f;
                st->custom_font = true;
                return;
            }
        }
    }
    st->ui_font = GetFontDefault();
    st->custom_font = false;
}

static void draw_txt(const GuiState *st, const char *text, float x, float y, float size, Color col) {
    DrawTextEx(st->ui_font, text, (Vector2){x, y}, size, 0.5f, col);
}

static int measure_txt_w(const GuiState *st, const char *text, float size) {
    Vector2 v = MeasureTextEx(st->ui_font, text, size, 0.5f);
    return (int)v.x;
}

static bool is_ai_turn_gui(GuiGameMode mode, Cell player) {
    if (mode == GUI_MODE_AI_VS_AI) return true;
    if (mode == GUI_MODE_HUMAN_BLACK) return player == CELL_WHITE;
    if (mode == GUI_MODE_HUMAN_WHITE) return player == CELL_BLACK;
    return false;
}

static void refresh_forbidden_cache(GuiState *st) {
    memset(st->forbidden_map, 0, sizeof(st->forbidden_map));
    memset(st->forbidden_reasons, 0, sizeof(st->forbidden_reasons));
    if (st->game_over || st->current_player != CELL_BLACK) {
        return;
    }
    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            if (!board_is_empty(&st->board, r, c)) {
                continue;
            }
            ForbiddenReason reason = rules_check_black_forbidden(&st->board, r, c);
            if (reason != FORBIDDEN_NONE) {
                int idx = r * BOARD_SIZE + c;
                st->forbidden_map[idx] = true;
                st->forbidden_reasons[idx] = reason;
            }
        }
    }
}

static void update_win_prob_from_score(GuiState *st, int score, Cell mover) {
    double side_p = 0.5 + 0.5 * tanh((double)score / 25000.0);
    if (score >= EVAL_MATE_SCORE - 1000) side_p = 0.99;
    if (score <= -(EVAL_MATE_SCORE - 1000)) side_p = 0.01;
    st->black_win_prob = (float)((mover == CELL_BLACK) ? side_p : (1.0 - side_p));
}

static void reset_game(GuiState *st) {
    board_init(&st->board);
    st->history_len = 0;
    st->current_player = CELL_BLACK;
    st->last_move = (Move){-1, -1};
    st->hint_move = (Move){-1, -1};
    st->has_hint = false;
    st->game_over = false;
    st->final_outcome = (MoveOutcome){.result = MOVE_OK, .forbidden = FORBIDDEN_NONE};
    st->black_win_prob = 0.52f;
    ++st->generation;
    gui_mutex_lock(&st->shared);
    st->shared.has_result = false;
    gui_mutex_unlock(&st->shared);
    snprintf(st->status_banner, sizeof(st->status_banner), "Black (X) to move");
    snprintf(st->last_ai_info, sizeof(st->last_ai_info), "Ready (d=%d, %.0fs)", st->max_depth, kTimeOptions[st->time_idx]);
    refresh_forbidden_cache(st);
}

static void apply_gui_move(
    GuiState *st,
    int row,
    int col,
    bool is_ai,
    const SearchStats *stats
) {
    Cell mover = st->current_player;
    MoveOutcome outcome = rules_apply_move(&st->board, row, col, mover);
    if (outcome.result == MOVE_INVALID) {
        snprintf(st->toast_msg, sizeof(st->toast_msg), "Illegal move at (%d, %d)", row + 1, col + 1);
        st->toast_until = GetTime() + 2.0;
        return;
    }

    st->last_move = (Move){row, col};
    st->has_hint = false;
    if (st->history_len < BOARD_CELLS) {
        GuiMoveRecord *rec = &st->history[st->history_len++];
        rec->move = st->last_move;
        rec->player = mover;
        rec->is_ai = is_ai;
        rec->depth = stats ? stats->completed_depth : 0;
        rec->elapsed_sec = stats ? stats->elapsed_sec : 0.0;
        rec->score = stats ? stats->score : 0;
        rec->used_vcf = stats ? stats->used_vcf : false;
        rec->used_guardrail = stats ? stats->used_guardrail : false;
    }

    if (is_ai && stats) {
        update_win_prob_from_score(st, stats->score, mover);
        snprintf(
            st->last_ai_info,
            sizeof(st->last_ai_info),
            "d=%d | %.0fms | %llu nodes%s%s",
            stats->completed_depth,
            stats->elapsed_sec * 1000.0,
            (unsigned long long)stats->nodes,
            stats->used_vcf ? " [VCF]" : "",
            stats->used_guardrail ? " [GUARD]" : ""
        );
    }

    if (outcome.result != MOVE_OK) {
        st->game_over = true;
        st->final_outcome = outcome;
        if (outcome.result == MOVE_BLACK_WIN) {
            st->black_win_prob = 1.0f;
            snprintf(st->status_banner, sizeof(st->status_banner), "Black (X) wins! (Five in a row)");
        } else if (outcome.result == MOVE_WHITE_WIN) {
            st->black_win_prob = 0.0f;
            snprintf(st->status_banner, sizeof(st->status_banner), "White (O) wins! (Five in a row)");
        } else if (outcome.result == MOVE_BLACK_FORBIDDEN) {
            st->black_win_prob = 0.0f;
            snprintf(
                st->status_banner,
                sizeof(st->status_banner),
                "White wins! Black forbidden (%s)",
                rules_forbidden_reason_name(outcome.forbidden)
            );
        } else {
            st->black_win_prob = 0.5f;
            snprintf(st->status_banner, sizeof(st->status_banner), "Draw (Board full)");
        }
    } else {
        st->current_player = board_opponent(mover);
        snprintf(
            st->status_banner,
            sizeof(st->status_banner),
            "%s to move",
            st->current_player == CELL_BLACK ? "Black (X)" : "White (O)"
        );
    }
    refresh_forbidden_cache(st);
}

static void perform_undo(GuiState *st) {
    int steps = (st->mode == GUI_MODE_TWO_PLAYER) ? 1 : 2;
    if (st->mode == GUI_MODE_AI_VS_AI) {
        steps = 1;
    }
    if (st->game_over && st->mode != GUI_MODE_TWO_PLAYER && st->history_len > 0) {
        Cell last_p = st->history[st->history_len - 1].player;
        bool last_was_human = !is_ai_turn_gui(st->mode, last_p);
        if (last_was_human) {
            steps = 1;
        }
    }
    if (st->history_len < steps || st->history_len == 0) {
        snprintf(st->toast_msg, sizeof(st->toast_msg), "Not enough moves to undo");
        st->toast_until = GetTime() + 1.8;
        return;
    }

    ++st->generation;
    gui_mutex_lock(&st->shared);
    st->shared.has_result = false;
    gui_mutex_unlock(&st->shared);

    for (int i = 0; i < steps; ++i) {
        --st->history_len;
        Move m = st->history[st->history_len].move;
        board_undo(&st->board, m.row, m.col);
    }

    st->game_over = false;
    st->has_hint = false;
    st->final_outcome = (MoveOutcome){.result = MOVE_OK, .forbidden = FORBIDDEN_NONE};
    st->last_move = st->history_len > 0
        ? st->history[st->history_len - 1].move
        : (Move){-1, -1};
    st->current_player = (st->history_len % 2 == 0) ? CELL_BLACK : CELL_WHITE;
    snprintf(
        st->status_banner,
        sizeof(st->status_banner),
        "Undid %d move(s) - %s to move",
        steps,
        st->current_player == CELL_BLACK ? "Black" : "White"
    );
    refresh_forbidden_cache(st);
}

static void save_gui_record(GuiState *st) {
    const char *path = "gomoku_gui_record.txt";
    FILE *fp = fopen(path, "w");
    if (!fp) {
        snprintf(st->toast_msg, sizeof(st->toast_msg), "Failed to save %s", path);
        st->toast_until = GetTime() + 2.5;
        return;
    }
    fprintf(fp, "# Gomoku AI GUI Record\n");
    fprintf(fp, "Total Plies: %d\n", st->history_len);
    fprintf(fp, "Status: %s\n\n", st->status_banner);
    fprintf(fp, "Ply | Player | Move (row,col) | Source | Depth | Time(s) | Score\n");
    fprintf(fp, "----+--------+----------------+--------+-------+---------+-------\n");
    for (int i = 0; i < st->history_len; ++i) {
        const GuiMoveRecord *r = &st->history[i];
        fprintf(
            fp,
            "%3d | %-6s | (%2d, %2d)       | %-6s | %5d | %7.3f | %6d%s%s\n",
            i + 1,
            r->player == CELL_BLACK ? "Black" : "White",
            r->move.row + 1,
            r->move.col + 1,
            r->is_ai ? "AI" : "Human",
            r->depth,
            r->elapsed_sec,
            r->score,
            r->used_vcf ? " [VCF]" : "",
            r->used_guardrail ? " [GUARD]" : ""
        );
    }
    fclose(fp);
    snprintf(st->toast_msg, sizeof(st->toast_msg), "Saved to %s", path);
    st->toast_until = GetTime() + 2.5;
}

static bool ui_button(
    const GuiState *st,
    Rectangle rect,
    const char *label,
    bool active,
    bool enabled,
    Color active_bg
) {
    Vector2 mouse = GetMousePosition();
    bool hover = enabled && CheckCollisionPointRec(mouse, rect);
    Color bg;
    if (!enabled) {
        bg = (Color){44, 49, 58, 255};
    } else if (active) {
        bg = active_bg;
    } else if (hover) {
        bg = (Color){68, 76, 92, 255};
    } else {
        bg = (Color){52, 58, 70, 255};
    }
    Color border = active ? (Color){145, 200, 255, 255} : (Color){82, 90, 108, 255};
    Color fg = enabled ? (Color){236, 240, 246, 255} : (Color){120, 128, 140, 255};

    DrawRectangleRounded(rect, 0.22f, 6, bg);
    DrawRectangleLinesEx(rect, active ? 1.5f : 1.0f, border);

    float fs = 15.0f;
    int tw = measure_txt_w(st, label, fs);
    draw_txt(st, label, rect.x + (rect.width - (float)tw) * 0.5f, rect.y + (rect.height - fs) * 0.5f, fs, fg);
    return enabled && hover && IsMouseButtonPressed(MOUSE_BUTTON_LEFT);
}

static void draw_stone(int cx, int cy, Cell color, bool is_last, int move_num, const GuiState *st) {
    float rad = (float)CELL_PX * 0.43f;
    DrawCircle(cx + 2, cy + 3, rad, (Color){0, 0, 0, 75});
    if (color == CELL_BLACK) {
        DrawCircleGradient(cx - 4, cy - 4, rad, (Color){82, 86, 94, 255}, (Color){18, 20, 24, 255});
        DrawCircleLines(cx, cy, rad, (Color){10, 10, 12, 255});
    } else {
        DrawCircleGradient(cx - 4, cy - 4, rad, (Color){255, 255, 255, 255}, (Color){206, 210, 218, 255});
        DrawCircleLines(cx, cy, rad, (Color){148, 152, 160, 255});
    }

    if (is_last) {
        Color ring_col = (color == CELL_BLACK) ? (Color){255, 196, 64, 255} : (Color){220, 64, 52, 255};
        DrawRing((Vector2){(float)cx, (float)cy}, rad * 0.34f, rad * 0.50f, 0.0f, 360.0f, 24, ring_col);
    }

    if (st->show_numbers && move_num > 0) {
        char buf[8];
        snprintf(buf, sizeof(buf), "%d", move_num);
        float fs = move_num >= 100 ? 11.0f : 13.0f;
        int tw = measure_txt_w(st, buf, fs);
        Color num_col = (color == CELL_BLACK) ? (Color){230, 234, 240, 235} : (Color){28, 32, 40, 235};
        draw_txt(st, buf, (float)cx - (float)tw * 0.5f, (float)cy - fs * 0.5f, fs, num_col);
    }
}

static void render_board(const GuiState *st, int hover_r, int hover_c) {
    Rectangle outer = {(float)BOARD_RECT_X, (float)BOARD_RECT_Y, (float)BOARD_RECT_SIZE, (float)BOARD_RECT_SIZE};
    DrawRectangleRounded((Rectangle){outer.x + 5, outer.y + 7, outer.width, outer.height}, 0.03f, 8, (Color){0, 0, 0, 90});
    DrawRectangleRounded(outer, 0.03f, 8, (Color){220, 179, 118, 255});
    DrawRectangleLinesEx(outer, 2.0f, (Color){132, 92, 44, 255});

    for (int i = 0; i < BOARD_SIZE; ++i) {
        int gx = GRID_ORIGIN_X + i * CELL_PX;
        int gy = GRID_ORIGIN_Y + i * CELL_PX;
        int end_x = GRID_ORIGIN_X + (BOARD_SIZE - 1) * CELL_PX;
        int end_y = GRID_ORIGIN_Y + (BOARD_SIZE - 1) * CELL_PX;
        float thick = (i == 0 || i == BOARD_SIZE - 1) ? 2.0f : 1.1f;
        DrawLineEx((Vector2){(float)GRID_ORIGIN_X, (float)gy}, (Vector2){(float)end_x, (float)gy}, thick, (Color){78, 52, 24, 235});
        DrawLineEx((Vector2){(float)gx, (float)GRID_ORIGIN_Y}, (Vector2){(float)gx, (float)end_y}, thick, (Color){78, 52, 24, 235});

        char lbl[8];
        snprintf(lbl, sizeof(lbl), "%d", i + 1);
        int tw = measure_txt_w(st, lbl, 13.0f);
        draw_txt(st, lbl, (float)gx - (float)tw * 0.5f, (float)GRID_ORIGIN_Y - 26.0f, 13.0f, (Color){76, 52, 26, 220});
        draw_txt(st, lbl, (float)GRID_ORIGIN_X - 28.0f - (float)(i >= 9 ? 6 : 0), (float)gy - 7.0f, 13.0f, (Color){76, 52, 26, 220});
    }

    static const int kStars[5][2] = {{3, 3}, {3, 11}, {7, 7}, {11, 3}, {11, 11}};
    for (int s = 0; s < 5; ++s) {
        int sx = GRID_ORIGIN_X + kStars[s][1] * CELL_PX;
        int sy = GRID_ORIGIN_Y + kStars[s][0] * CELL_PX;
        DrawCircle(sx, sy, 4.3f, (Color){62, 40, 18, 245});
    }

    int move_num_map[BOARD_CELLS];
    memset(move_num_map, 0, sizeof(move_num_map));
    for (int i = 0; i < st->history_len; ++i) {
        Move m = st->history[i].move;
        move_num_map[m.row * BOARD_SIZE + m.col] = i + 1;
    }

    if (st->show_forbidden && !st->game_over && st->current_player == CELL_BLACK) {
        for (int r = 0; r < BOARD_SIZE; ++r) {
            for (int c = 0; c < BOARD_SIZE; ++c) {
                if (st->forbidden_map[r * BOARD_SIZE + c]) {
                    int cx = GRID_ORIGIN_X + c * CELL_PX;
                    int cy = GRID_ORIGIN_Y + r * CELL_PX;
                    DrawLineEx((Vector2){(float)(cx - 7), (float)(cy - 7)}, (Vector2){(float)(cx + 7), (float)(cy + 7)}, 2.2f, (Color){205, 42, 42, 220});
                    DrawLineEx((Vector2){(float)(cx + 7), (float)(cy - 7)}, (Vector2){(float)(cx - 7), (float)(cy + 7)}, 2.2f, (Color){205, 42, 42, 220});
                }
            }
        }
    }

    for (int r = 0; r < BOARD_SIZE; ++r) {
        for (int c = 0; c < BOARD_SIZE; ++c) {
            Cell cell = st->board.cells[r][c];
            if (cell == CELL_EMPTY) continue;
            int cx = GRID_ORIGIN_X + c * CELL_PX;
            int cy = GRID_ORIGIN_Y + r * CELL_PX;
            bool is_last = (r == st->last_move.row && c == st->last_move.col);
            draw_stone(cx, cy, cell, is_last, move_num_map[r * BOARD_SIZE + c], st);
        }
    }

    if (st->has_hint && !st->game_over) {
        int hx = GRID_ORIGIN_X + st->hint_move.col * CELL_PX;
        int hy = GRID_ORIGIN_Y + st->hint_move.row * CELL_PX;
        DrawRing((Vector2){(float)hx, (float)hy}, 10.0f, 15.0f, 0.0f, 360.0f, 24, (Color){36, 130, 235, 240});
        DrawCircle(hx, hy, 5.0f, (Color){36, 130, 235, 200});
    }

    if (!st->game_over && !is_ai_turn_gui(st->mode, st->current_player)
        && hover_r >= 0 && hover_c >= 0
        && board_is_empty(&st->board, hover_r, hover_c)) {
        int cx = GRID_ORIGIN_X + hover_c * CELL_PX;
        int cy = GRID_ORIGIN_Y + hover_r * CELL_PX;
        Color ghost = (st->current_player == CELL_BLACK)
            ? (Color){20, 22, 26, 95}
            : (Color){250, 252, 255, 130};
        DrawCircle(cx, cy, (float)CELL_PX * 0.40f, ghost);
        if (st->current_player == CELL_BLACK && st->forbidden_map[hover_r * BOARD_SIZE + hover_c]) {
            DrawRing((Vector2){(float)cx, (float)cy}, 14.0f, 18.0f, 0.0f, 360.0f, 24, (Color){220, 48, 48, 220});
        }
    }
}

int main(void) {
    SetConfigFlags(FLAG_WINDOW_HIGHDPI | FLAG_MSAA_4X_HINT);
    InitWindow(GUI_WIDTH, GUI_HEIGHT, "Gomoku AI - Pattern-NNUE & Renju (15x15)");
    SetTargetFPS(60);

    GuiState st;
    memset(&st, 0, sizeof(st));
    gui_mutex_init(&st.shared);
    load_best_font(&st);

    char pnnue_path[1024] = {0};
    char nnue_path[1024] = {0};
    char cnn_path[1024] = {0};
    if (resolve_model_file("pnnue.bin", pnnue_path, sizeof(pnnue_path))) {
        st.pnnue = pnnue_load(pnnue_path);
    }
    if (resolve_model_file("nnue.bin", nnue_path, sizeof(nnue_path))) {
        st.nnue = nnue_load(nnue_path);
    }
    if (resolve_model_file("cnn.bin", cnn_path, sizeof(cnn_path))) {
        st.cnn = cnn_load(cnn_path);
    }

    if (st.pnnue) {
        st.engine = ENGINE_V3_PNNUE;
    } else if (st.nnue || st.cnn) {
        st.engine = ENGINE_V2_DUAL;
    } else {
        st.engine = ENGINE_HEURISTIC;
    }

    printf(
        "Gomoku GUI models: V3=%s | V2-NNUE=%s | V2-CNN=%s\n",
        st.pnnue ? pnnue_path : "(none)",
        st.nnue ? nnue_path : "(none)",
        st.cnn ? cnn_path : "(none)"
    );
    fflush(stdout);

    st.mode = GUI_MODE_HUMAN_BLACK;
    st.max_depth = 5;
    st.time_idx = 1;
    st.show_forbidden = true;
    st.show_numbers = false;
    reset_game(&st);

    while (!WindowShouldClose()) {
        /* 1. Consume completed AI worker result */
        bool got_res = false;
        bool res_is_hint = false;
        uint64_t res_gen = 0;
        SearchResult res;
        memset(&res, 0, sizeof(res));

        gui_mutex_lock(&st.shared);
        bool ai_busy = st.shared.busy;
        if (st.shared.has_result) {
            got_res = true;
            res_gen = st.shared.result_gen;
            res_is_hint = st.shared.result_is_hint;
            res = st.shared.result;
            st.shared.has_result = false;
        }
        gui_mutex_unlock(&st.shared);

        if (got_res && res_gen == st.generation && !st.game_over && res.valid) {
            if (res_is_hint) {
                st.hint_move = res.best_move;
                st.has_hint = true;
                update_win_prob_from_score(&st, res.stats.score, st.current_player);
                snprintf(
                    st.last_ai_info,
                    sizeof(st.last_ai_info),
                    "Hint (%d,%d) d=%d %.0fms",
                    res.best_move.row + 1,
                    res.best_move.col + 1,
                    res.stats.completed_depth,
                    res.stats.elapsed_sec * 1000.0
                );
            } else {
                apply_gui_move(&st, res.best_move.row, res.best_move.col, true, &res.stats);
            }
        }

        /* 2. Auto-trigger AI turn when idle */
        if (!st.game_over && !ai_busy && is_ai_turn_gui(st.mode, st.current_player)) {
            if (spawn_ai_worker(&st, false)) {
                ai_busy = true;
            }
        }

        /* 3. Board hover / click input */
        Vector2 mouse = GetMousePosition();
        int hover_c = (int)floorf((mouse.x - (float)GRID_ORIGIN_X) / (float)CELL_PX + 0.5f);
        int hover_r = (int)floorf((mouse.y - (float)GRID_ORIGIN_Y) / (float)CELL_PX + 0.5f);
        if (mouse.x < BOARD_RECT_X || mouse.x > BOARD_RECT_X + BOARD_RECT_SIZE ||
            mouse.y < BOARD_RECT_Y || mouse.y > BOARD_RECT_Y + BOARD_RECT_SIZE ||
            hover_r < 0 || hover_r >= BOARD_SIZE || hover_c < 0 || hover_c >= BOARD_SIZE) {
            hover_r = -1;
            hover_c = -1;
        }

        if (!st.game_over && !ai_busy && !is_ai_turn_gui(st.mode, st.current_player)
            && hover_r >= 0 && hover_c >= 0 && IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
            if (board_is_empty(&st.board, hover_r, hover_c)) {
                apply_gui_move(&st, hover_r, hover_c, false, NULL);
            }
        }

        /* 4. Render frame */
        BeginDrawing();
        ClearBackground((Color){24, 28, 36, 255});

        render_board(&st, hover_r, hover_c);

        /* Sidebar panel */
        Rectangle panel = {(float)SIDEBAR_X, (float)SIDEBAR_Y, (float)SIDEBAR_W, (float)SIDEBAR_H};
        DrawRectangleRounded(panel, 0.03f, 8, (Color){34, 38, 48, 255});
        DrawRectangleLinesEx(panel, 1.5f, (Color){64, 72, 90, 255});

        float x = (float)SIDEBAR_X + 16.0f;
        float w = (float)SIDEBAR_W - 32.0f;
        float y = (float)SIDEBAR_Y + 16.0f;

        draw_txt(&st, "Gomoku AI Studio", x, y, 22.0f, (Color){242, 245, 250, 255});
        y += 28.0f;
        const char *eng_lbl = (st.engine == ENGINE_V3_PNNUE)
            ? "Active: V3 Pattern-NNUE (GPN1)"
            : (st.engine == ENGINE_V2_DUAL ? "Active: V2+Rapfi Dual-Net" : "Active: Pure Heuristic");
        draw_txt(&st, eng_lbl, x, y, 14.0f, (Color){126, 210, 176, 255});
        y += 26.0f;

        /* Engine selector */
        draw_txt(&st, "Engine", x, y, 14.0f, (Color){168, 178, 196, 255});
        y += 20.0f;
        float bw3 = (w - 12.0f) / 3.0f;
        Color blue_hi = (Color){56, 118, 206, 255};
        if (ui_button(&st, (Rectangle){x, y, bw3, 30.0f}, "V3 PNNUE", st.engine == ENGINE_V3_PNNUE, st.pnnue != NULL, blue_hi)) {
            st.engine = ENGINE_V3_PNNUE;
        }
        if (ui_button(&st, (Rectangle){x + bw3 + 6.0f, y, bw3, 30.0f}, "V2 Dual", st.engine == ENGINE_V2_DUAL, (st.nnue != NULL || st.cnn != NULL), blue_hi)) {
            st.engine = ENGINE_V2_DUAL;
        }
        if (ui_button(&st, (Rectangle){x + 2.0f * (bw3 + 6.0f), y, bw3, 30.0f}, "Heur", st.engine == ENGINE_HEURISTIC, true, blue_hi)) {
            st.engine = ENGINE_HEURISTIC;
        }
        y += 42.0f;

        /* Mode selector */
        draw_txt(&st, "Game Mode", x, y, 14.0f, (Color){168, 178, 196, 255});
        y += 20.0f;
        float bw2 = (w - 8.0f) * 0.5f;
        if (ui_button(&st, (Rectangle){x, y, bw2, 30.0f}, "Human Black", st.mode == GUI_MODE_HUMAN_BLACK, true, blue_hi)) {
            st.mode = GUI_MODE_HUMAN_BLACK;
            reset_game(&st);
        }
        if (ui_button(&st, (Rectangle){x + bw2 + 8.0f, y, bw2, 30.0f}, "Human White", st.mode == GUI_MODE_HUMAN_WHITE, true, blue_hi)) {
            st.mode = GUI_MODE_HUMAN_WHITE;
            reset_game(&st);
        }
        y += 36.0f;
        if (ui_button(&st, (Rectangle){x, y, bw2, 30.0f}, "2 Players", st.mode == GUI_MODE_TWO_PLAYER, true, blue_hi)) {
            st.mode = GUI_MODE_TWO_PLAYER;
            reset_game(&st);
        }
        if (ui_button(&st, (Rectangle){x + bw2 + 8.0f, y, bw2, 30.0f}, "AI vs AI", st.mode == GUI_MODE_AI_VS_AI, true, blue_hi)) {
            st.mode = GUI_MODE_AI_VS_AI;
            reset_game(&st);
        }
        y += 44.0f;

        /* Depth selector */
        draw_txt(&st, "Search Depth", x, y, 14.0f, (Color){168, 178, 196, 255});
        y += 20.0f;
        float bw4 = (w - 18.0f) / 4.0f;
        for (int i = 0; i < 4; ++i) {
            char dlbl[16];
            snprintf(dlbl, sizeof(dlbl), "d=%d", kDepthOptions[i]);
            if (ui_button(&st, (Rectangle){x + (float)i * (bw4 + 6.0f), y, bw4, 28.0f}, dlbl, st.max_depth == kDepthOptions[i], true, blue_hi)) {
                st.max_depth = kDepthOptions[i];
            }
        }
        y += 40.0f;

        /* Time limit selector */
        draw_txt(&st, "Time Budget / Move", x, y, 14.0f, (Color){168, 178, 196, 255});
        y += 20.0f;
        for (int i = 0; i < 3; ++i) {
            char tlbl[16];
            snprintf(tlbl, sizeof(tlbl), "%.0fs", kTimeOptions[i]);
            if (ui_button(&st, (Rectangle){x + (float)i * (bw3 + 6.0f), y, bw3, 28.0f}, tlbl, st.time_idx == i, true, blue_hi)) {
                st.time_idx = i;
            }
        }
        y += 44.0f;

        /* Action buttons */
        Color green_hi = (Color){44, 138, 92, 255};
        if (ui_button(&st, (Rectangle){x, y, bw2, 34.0f}, "New Game", false, true, green_hi)) {
            reset_game(&st);
        }
        if (ui_button(&st, (Rectangle){x + bw2 + 8.0f, y, bw2, 34.0f}, "Undo", false, st.history_len > 0, green_hi)) {
            perform_undo(&st);
        }
        y += 42.0f;
        bool can_hint = !st.game_over && !ai_busy && !is_ai_turn_gui(st.mode, st.current_player);
        if (ui_button(&st, (Rectangle){x, y, bw2, 34.0f}, "AI Hint", st.has_hint, can_hint, blue_hi)) {
            spawn_ai_worker(&st, true);
        }
        if (ui_button(&st, (Rectangle){x + bw2 + 8.0f, y, bw2, 34.0f}, "Save Record", false, st.history_len > 0, green_hi)) {
            save_gui_record(&st);
        }
        y += 46.0f;

        /* Display toggles */
        Color teal_hi = (Color){42, 128, 138, 255};
        if (ui_button(&st, (Rectangle){x, y, bw2, 28.0f}, "Forbidden X", st.show_forbidden, true, teal_hi)) {
            st.show_forbidden = !st.show_forbidden;
        }
        if (ui_button(&st, (Rectangle){x + bw2 + 8.0f, y, bw2, 28.0f}, "Move #", st.show_numbers, true, teal_hi)) {
            st.show_numbers = !st.show_numbers;
        }
        y += 44.0f;

        /* Telemetry card */
        Rectangle card = {x, y, w, 198.0f};
        DrawRectangleRounded(card, 0.08f, 6, (Color){24, 28, 36, 255});
        DrawRectangleLinesEx(card, 1.0f, (Color){64, 72, 88, 255});

        float cy = y + 12.0f;
        Color status_col = st.game_over ? (Color){250, 206, 92, 255} : (Color){228, 234, 244, 255};
        draw_txt(&st, st.status_banner, x + 12.0f, cy, 16.0f, status_col);
        cy += 26.0f;

        char ply_line[64];
        if (ai_busy) {
            snprintf(ply_line, sizeof(ply_line), "AI Thinking... %.1fs (Ply %d)", GetTime() - st.think_start_sec, st.history_len);
        } else {
            snprintf(ply_line, sizeof(ply_line), "Ply: %d / %d", st.history_len, BOARD_CELLS);
        }
        draw_txt(&st, ply_line, x + 12.0f, cy, 14.0f, ai_busy ? (Color){120, 196, 255, 255} : (Color){172, 182, 200, 255});
        cy += 26.0f;

        /* Win-rate bar */
        char wr_lbl[64];
        snprintf(wr_lbl, sizeof(wr_lbl), "Win Prob  B %.0f%%  /  W %.0f%%", st.black_win_prob * 100.0f, (1.0f - st.black_win_prob) * 100.0f);
        draw_txt(&st, wr_lbl, x + 12.0f, cy, 13.0f, (Color){184, 192, 208, 255});
        cy += 18.0f;
        float bar_w = w - 24.0f;
        DrawRectangleRounded((Rectangle){x + 12.0f, cy, bar_w, 12.0f}, 0.5f, 4, (Color){224, 228, 236, 255});
        float bw_fill = bar_w * st.black_win_prob;
        if (bw_fill > 4.0f) {
            DrawRectangleRounded((Rectangle){x + 12.0f, cy, bw_fill, 12.0f}, 0.5f, 4, (Color){44, 48, 56, 255});
        }
        DrawRectangleLinesEx((Rectangle){x + 12.0f, cy, bar_w, 12.0f}, 1.0f, (Color){92, 100, 118, 255});
        cy += 24.0f;

        draw_txt(&st, st.last_ai_info, x + 12.0f, cy, 13.0f, (Color){144, 202, 168, 255});
        cy += 24.0f;

        if (hover_r >= 0 && hover_c >= 0) {
            char coord_buf[64];
            if (!st.game_over && st.current_player == CELL_BLACK && st.forbidden_map[hover_r * BOARD_SIZE + hover_c]) {
                snprintf(
                    coord_buf,
                    sizeof(coord_buf),
                    "Cursor: (%d, %d) FORBIDDEN (%s)",
                    hover_r + 1,
                    hover_c + 1,
                    rules_forbidden_reason_name(st.forbidden_reasons[hover_r * BOARD_SIZE + hover_c])
                );
                draw_txt(&st, coord_buf, x + 12.0f, cy, 13.0f, (Color){236, 98, 98, 255});
            } else {
                snprintf(coord_buf, sizeof(coord_buf), "Cursor: (%d, %d)", hover_r + 1, hover_c + 1);
                draw_txt(&st, coord_buf, x + 12.0f, cy, 13.0f, (Color){156, 166, 184, 255});
            }
        } else {
            draw_txt(&st, "Click intersection to place stone", x + 12.0f, cy, 13.0f, (Color){132, 142, 160, 255});
        }

        if (GetTime() < st.toast_until) {
            draw_txt(&st, st.toast_msg, x + 12.0f, y + 206.0f, 14.0f, (Color){126, 224, 168, 255});
        }

        EndDrawing();
    }

    if (st.custom_font) {
        UnloadFont(st.ui_font);
    }
    CloseWindow();

    /* Let background search finish before freeing models */
    for (;;) {
        gui_mutex_lock(&st.shared);
        bool busy = st.shared.busy;
        gui_mutex_unlock(&st.shared);
        if (!busy) break;
#if defined(_WIN32)
        Sleep(10);
#endif
    }

    pnnue_free(st.pnnue);
    nnue_free(st.nnue);
    cnn_free(st.cnn);
    gui_mutex_destroy(&st.shared);
    return 0;
}
