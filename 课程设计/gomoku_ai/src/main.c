#include "board.h"
#include "console_ui.h"
#include "nn_model.h"
#include "rules.h"
#include "search.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    bool ai_vs_ai;
    int games;
    double time_limit;
    int max_depth;
    const char *nnue_path;
    const char *cnn_path;
    const char *save_record_path;
    bool no_nn;
    bool show_help;
} CliOptions;

typedef struct {
    Move move;
    Cell player;
    bool is_ai;
    int depth;
    double elapsed_sec;
    int score;
    bool used_vcf;
    bool used_guardrail;
} GameMoveRecord;

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --ai-vs-ai          Run automated AI vs AI matches without menu\n");
    printf("  --games N           Number of games for --ai-vs-ai (default: 1)\n");
    printf("  --time SEC          Search time limit per move in seconds (default: 5.0)\n");
    printf("  --depth D           Max search depth (default: 5)\n");
    printf("  --nnue PATH         Path to NNUE-Lite model binary (default: models/nnue.bin)\n");
    printf("  --cnn PATH          Path to Policy-Value CNN binary (default: models/cnn.bin)\n");
    printf("  --save-record PATH  Save replayable move log to PATH\n");
    printf("  --no-nn             Disable neural network models and use linear heuristic\n");
    printf("  --help              Show this help\n");
}

static bool parse_cli(int argc, char **argv, CliOptions *opts) {
    opts->ai_vs_ai = false;
    opts->games = 1;
    opts->time_limit = 5.0;
    opts->max_depth = 5;
    opts->nnue_path = "models/nnue.bin";
    opts->cnn_path = "models/cnn.bin";
    opts->save_record_path = NULL;
    opts->no_nn = false;
    opts->show_help = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--ai-vs-ai") == 0) {
            opts->ai_vs_ai = true;
        } else if (strcmp(argv[i], "--games") == 0 && i + 1 < argc) {
            opts->games = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--time") == 0 && i + 1 < argc) {
            opts->time_limit = atof(argv[++i]);
        } else if (strcmp(argv[i], "--depth") == 0 && i + 1 < argc) {
            opts->max_depth = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nnue") == 0 && i + 1 < argc) {
            opts->nnue_path = argv[++i];
        } else if (strcmp(argv[i], "--cnn") == 0 && i + 1 < argc) {
            opts->cnn_path = argv[++i];
        } else if (strcmp(argv[i], "--save-record") == 0 && i + 1 < argc) {
            opts->save_record_path = argv[++i];
        } else if (strcmp(argv[i], "--no-nn") == 0) {
            opts->no_nn = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            opts->show_help = true;
            return true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return false;
        }
    }

    if (opts->games < 1) opts->games = 1;
    if (opts->time_limit <= 0.0) opts->time_limit = 1.0;
    if (opts->max_depth < 1) opts->max_depth = 1;
    return true;
}

static bool is_ai_turn(GameMode mode, Cell player) {
    if (mode == GAME_MODE_AI_VS_AI) return true;
    if (mode == GAME_MODE_HUMAN_BLACK) return player == CELL_WHITE;
    if (mode == GAME_MODE_HUMAN_WHITE) return player == CELL_BLACK;
    return false;
}

static const char *mode_name(GameMode mode) {
    switch (mode) {
        case GAME_MODE_HUMAN_BLACK: return "Human (Black) vs AI (White)";
        case GAME_MODE_HUMAN_WHITE: return "AI (Black) vs Human (White)";
        case GAME_MODE_TWO_PLAYER:  return "Human vs Human";
        case GAME_MODE_AI_VS_AI:    return "AI vs AI";
        default:                    return "Unknown";
    }
}

static void save_game_record(
    const char *path,
    GameMode mode,
    const Board *board,
    const GameMoveRecord *history,
    int history_len,
    MoveOutcome final_outcome,
    bool game_finished
) {
    if (!path || path[0] == '\0') {
        return;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Warning: could not open record file '%s' for writing.\n", path);
        return;
    }

    fprintf(fp, "# Gomoku AI Game Record\n");
    fprintf(fp, "Mode: %s\n", mode_name(mode));
    fprintf(fp, "Board: %dx%d (Black forbidden rules enabled)\n", BOARD_SIZE, BOARD_SIZE);
    fprintf(fp, "Total Plies: %d\n", history_len);
    if (game_finished) {
        const char *res_str = "Draw";
        if (final_outcome.result == MOVE_BLACK_WIN) res_str = "Black (X) wins";
        else if (final_outcome.result == MOVE_WHITE_WIN) res_str = "White (O) wins";
        else if (final_outcome.result == MOVE_BLACK_FORBIDDEN) res_str = "Black loses by forbidden move";
        fprintf(fp, "Result: %s\n", res_str);
    } else {
        fprintf(fp, "Result: Ended before completion\n");
    }

    fprintf(fp, "\nPly | Player | Move (row,col) | Source | Depth | Time(s) | Score\n");
    fprintf(fp, "----+--------+----------------+--------+-------+---------+-------\n");
    for (int i = 0; i < history_len; ++i) {
        const GameMoveRecord *rec = &history[i];
        fprintf(
            fp,
            "%3d | %-6s | (%2d, %2d)       | %-6s | %5d | %7.3f | %6d%s%s\n",
            i + 1,
            rec->player == CELL_BLACK ? "Black" : "White",
            rec->move.row + 1,
            rec->move.col + 1,
            rec->is_ai ? "AI" : "Human",
            rec->depth,
            rec->elapsed_sec,
            rec->score,
            rec->used_vcf ? " [VCF]" : "",
            rec->used_guardrail ? " [GUARD]" : ""
        );
    }

    fprintf(fp, "\nFinal Board:\n    ");
    for (int c = 0; c < BOARD_SIZE; ++c) {
        fprintf(fp, "%2d ", c + 1);
    }
    fprintf(fp, "\n");
    for (int r = 0; r < BOARD_SIZE; ++r) {
        fprintf(fp, "%2d  ", r + 1);
        for (int c = 0; c < BOARD_SIZE; ++c) {
            Cell cell = board->cells[r][c];
            char ch = cell == CELL_BLACK ? 'X' : (cell == CELL_WHITE ? 'O' : '.');
            fprintf(fp, " %c ", ch);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Saved replayable game record to %s\n", path);
}

static int run_single_game(
    GameMode mode,
    const SearchConfig *cfg,
    const NnueModel *nnue,
    const CnnModel *cnn,
    const char *record_path,
    bool quiet_board
) {
    Board board;
    board_init(&board);

    GameMoveRecord history[BOARD_CELLS];
    int history_len = 0;
    Cell current_player = CELL_BLACK;
    Move last_move = {-1, -1};
    MoveOutcome last_outcome = {.result = MOVE_OK, .forbidden = FORBIDDEN_NONE};
    bool game_finished = false;
    int rc = 0;

    for (;;) {
        if (!quiet_board) {
            ui_print_board_with_last(&board, last_move);
        }

        int row = -1;
        int col = -1;
        SearchStats ai_stats;
        memset(&ai_stats, 0, sizeof(ai_stats));
        bool ai_move = is_ai_turn(mode, current_player);

        if (ai_move) {
            SearchResult res = search_find_best_move(
                &board, current_player, last_move, cfg, nnue, cnn
            );
            if (!res.valid) {
                fprintf(stderr, "AI could not find a legal move.\n");
                rc = -1;
                break;
            }
            row = res.best_move.row;
            col = res.best_move.col;
            ai_stats = res.stats;
            if (!quiet_board) {
                ui_print_ai_move(current_player, res.best_move, &res.stats);
            }
        } else {
            ui_prompt_move(current_player);
            InputResult input = ui_read_move(&row, &col);
            if (input == INPUT_QUIT) {
                printf("Game ended by player.\n");
                break;
            }
            if (input == INPUT_EOF) {
                printf("Input ended. Game stopped.\n");
                break;
            }
            if (input == INPUT_UNDO) {
                int steps_to_undo = (mode == GAME_MODE_TWO_PLAYER) ? 1 : 2;
                if (history_len < steps_to_undo) {
                    printf("Not enough moves played yet to undo.\n");
                    continue;
                }
                for (int u = 0; u < steps_to_undo; ++u) {
                    --history_len;
                    board_undo(&board, history[history_len].move.row, history[history_len].move.col);
                }
                last_move = history_len > 0
                    ? history[history_len - 1].move
                    : (Move){-1, -1};
                if (steps_to_undo == 1) {
                    current_player = board_opponent(current_player);
                }
                printf("Undid %d move(s).\n", steps_to_undo);
                continue;
            }
            if (input == INPUT_INVALID) {
                continue;
            }
        }

        MoveOutcome outcome = rules_apply_move(&board, row, col, current_player);
        if (outcome.result == MOVE_INVALID) {
            ui_print_move_error(outcome);
            if (ai_move) {
                fprintf(stderr, "Fatal: AI attempted invalid move!\n");
                rc = -1;
                break;
            }
            continue;
        }

        last_move.row = row;
        last_move.col = col;
        if (history_len < BOARD_CELLS) {
            history[history_len].move = last_move;
            history[history_len].player = current_player;
            history[history_len].is_ai = ai_move;
            history[history_len].depth = ai_stats.completed_depth;
            history[history_len].elapsed_sec = ai_stats.elapsed_sec;
            history[history_len].score = ai_stats.score;
            history[history_len].used_vcf = ai_stats.used_vcf;
            history[history_len].used_guardrail = ai_stats.used_guardrail;
            ++history_len;
        }

        if (outcome.result != MOVE_OK) {
            last_outcome = outcome;
            game_finished = true;
            if (!quiet_board || outcome.result == MOVE_BLACK_FORBIDDEN) {
                ui_print_board_with_last(&board, last_move);
            }
            ui_print_game_result(outcome);
            printf("Game finished in %d plies.\n", history_len);
            if (outcome.result == MOVE_BLACK_FORBIDDEN || outcome.result == MOVE_WHITE_WIN) {
                rc = 2;
            } else if (outcome.result == MOVE_BLACK_WIN) {
                rc = 1;
            } else {
                rc = 0;
            }
            break;
        }

        current_player = board_opponent(current_player);
    }

    if (record_path && history_len > 0) {
        save_game_record(
            record_path, mode, &board, history, history_len, last_outcome, game_finished
        );
    }
    return rc;
}

int main(int argc, char **argv) {
    CliOptions opts;
    if (!parse_cli(argc, argv, &opts)) {
        print_usage(argv[0]);
        return 1;
    }
    if (opts.show_help) {
        print_usage(argv[0]);
        return 0;
    }

    NnueModel *nnue = NULL;
    CnnModel *cnn = NULL;
    if (!opts.no_nn) {
        nnue = nnue_load(opts.nnue_path);
        cnn = cnn_load(opts.cnn_path);
        printf(
            "Engine models: NNUE=%s, CNN=%s\n",
            nnue ? opts.nnue_path : "none (linear fallback)",
            cnn ? opts.cnn_path : "none (heuristic ordering)"
        );
    } else {
        printf("Engine models: disabled (--no-nn, linear heuristic mode)\n");
    }

    SearchConfig cfg = search_default_config();
    cfg.time_limit_sec = opts.time_limit;
    cfg.max_depth = opts.max_depth;
    cfg.use_nnue = (nnue != NULL);
    cfg.use_cnn = (cnn != NULL);

    int rc = 0;
    if (opts.ai_vs_ai) {
        int black_wins = 0;
        int white_wins = 0;
        int draws = 0;
        for (int g = 0; g < opts.games; ++g) {
            printf("=== AI vs AI Game %d/%d ===\n", g + 1, opts.games);
            const char *rec_path = (g == opts.games - 1) ? opts.save_record_path : NULL;
            int res = run_single_game(
                GAME_MODE_AI_VS_AI, &cfg, nnue, cnn, rec_path, opts.games > 1
            );
            if (res < 0) {
                rc = 1;
                break;
            }
            if (res == 1) ++black_wins;
            else if (res == 2) ++white_wins;
            else ++draws;
        }
        if (rc == 0) {
            printf(
                "Summary (%d games): Black=%d, White=%d, Draw=%d\n",
                opts.games, black_wins, white_wins, draws
            );
        }
    } else {
        GameMode mode = GAME_MODE_HUMAN_BLACK;
        for (;;) {
            ui_prompt_mode();
            if (ui_read_mode(&mode)) {
                break;
            }
            if (feof(stdin)) {
                nnue_free(nnue);
                cnn_free(cnn);
                return 0;
            }
            printf("Invalid selection. Please enter 1, 2, 3, or 4.\n");
        }
        const char *rec_path = opts.save_record_path ? opts.save_record_path : "last_game.log";
        int res = run_single_game(mode, &cfg, nnue, cnn, rec_path, false);
        rc = res < 0 ? 1 : 0;
    }

    nnue_free(nnue);
    cnn_free(cnn);
    return rc;
}
