#include "board.h"
#include "console_ui.h"
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
    bool show_help;
} CliOptions;

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --ai-vs-ai       Run automated AI vs AI matches without menu\n");
    printf("  --games N        Number of games for --ai-vs-ai (default: 1)\n");
    printf("  --time SEC       Search time limit per move in seconds (default: 5.0)\n");
    printf("  --depth D        Max search depth (default: 4)\n");
    printf("  --nnue PATH      Path to NNUE-Lite model binary (optional)\n");
    printf("  --cnn PATH       Path to Policy-Value CNN model binary (optional)\n");
    printf("  --help           Show this help\n");
}

static bool parse_cli(int argc, char **argv, CliOptions *opts) {
    opts->ai_vs_ai = false;
    opts->games = 1;
    opts->time_limit = 5.0;
    opts->max_depth = 4;
    opts->nnue_path = "models/nnue.bin";
    opts->cnn_path = "models/cnn.bin";
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
    if (mode == GAME_MODE_AI_VS_AI) {
        return true;
    }
    if (mode == GAME_MODE_HUMAN_BLACK) {
        return player == CELL_WHITE;
    }
    if (mode == GAME_MODE_HUMAN_WHITE) {
        return player == CELL_BLACK;
    }
    return false;
}

static int run_single_game(
    GameMode mode,
    const SearchConfig *cfg,
    bool quiet_board
) {
    Board board;
    board_init(&board);

    Cell current_player = CELL_BLACK;
    Move last_move = {-1, -1};
    int ply = 0;

    for (;;) {
        if (!quiet_board) {
            ui_print_board_with_last(&board, last_move);
        }

        int row = -1;
        int col = -1;

        if (is_ai_turn(mode, current_player)) {
            SearchResult res = search_find_best_move(
                &board, current_player, last_move, cfg, NULL, NULL
            );
            if (!res.valid) {
                fprintf(stderr, "AI could not find a legal move.\n");
                return -1;
            }
            row = res.best_move.row;
            col = res.best_move.col;
            if (!quiet_board) {
                ui_print_ai_move(current_player, res.best_move, &res.stats);
            }
        } else {
            ui_prompt_move(current_player);
            InputResult input = ui_read_move(&row, &col);
            if (input == INPUT_QUIT) {
                printf("Game ended by player.\n");
                return 0;
            }
            if (input == INPUT_EOF) {
                printf("Input ended. Game stopped.\n");
                return 0;
            }
            if (input == INPUT_INVALID) {
                continue;
            }
        }

        MoveOutcome outcome = rules_apply_move(&board, row, col, current_player);
        if (outcome.result == MOVE_INVALID) {
            ui_print_move_error(outcome);
            if (is_ai_turn(mode, current_player)) {
                fprintf(stderr, "Fatal: AI attempted invalid move!\n");
                return -1;
            }
            continue;
        }

        last_move.row = row;
        last_move.col = col;
        ++ply;

        if (outcome.result != MOVE_OK) {
            if (!quiet_board || outcome.result == MOVE_BLACK_FORBIDDEN) {
                ui_print_board_with_last(&board, last_move);
            }
            ui_print_game_result(outcome);
            printf("Game finished in %d plies.\n", ply);
            if (outcome.result == MOVE_BLACK_FORBIDDEN) {
                return 2;
            }
            return outcome.result == MOVE_BLACK_WIN ? 1 :
                   (outcome.result == MOVE_WHITE_WIN ? 2 : 0);
        }

        current_player = board_opponent(current_player);
    }
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

    (void)opts.nnue_path;
    (void)opts.cnn_path;

    SearchConfig cfg = search_default_config();
    cfg.time_limit_sec = opts.time_limit;
    cfg.max_depth = opts.max_depth;

    if (opts.ai_vs_ai) {
        int black_wins = 0;
        int white_wins = 0;
        int draws = 0;
        for (int g = 0; g < opts.games; ++g) {
            printf("=== AI vs AI Game %d/%d ===\n", g + 1, opts.games);
            int res = run_single_game(GAME_MODE_AI_VS_AI, &cfg, opts.games > 1);
            if (res < 0) {
                return 1;
            }
            if (res == 1) ++black_wins;
            else if (res == 2) ++white_wins;
            else ++draws;
        }
        printf(
            "Summary (%d games): Black=%d, White=%d, Draw=%d\n",
            opts.games, black_wins, white_wins, draws
        );
        return 0;
    }

    GameMode mode = GAME_MODE_HUMAN_BLACK;
    for (;;) {
        ui_prompt_mode();
        if (ui_read_mode(&mode)) {
            break;
        }
        if (feof(stdin)) {
            return 0;
        }
        printf("Invalid selection. Please enter 1, 2, 3, or 4.\n");
    }

    int rc = run_single_game(mode, &cfg, false);
    return rc < 0 ? 1 : 0;
}
