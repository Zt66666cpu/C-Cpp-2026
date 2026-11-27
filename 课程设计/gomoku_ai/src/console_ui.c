#include "console_ui.h"

#include <stdio.h>
#include <string.h>

static char cell_symbol(Cell cell, bool is_last) {
    switch (cell) {
        case CELL_BLACK:
            return is_last ? '#' : 'X';
        case CELL_WHITE:
            return is_last ? '@' : 'O';
        case CELL_EMPTY:
        default:
            return '.';
    }
}

void ui_print_board_with_last(const Board *board, Move last_move) {
    printf("\n    ");
    for (int col = 0; col < BOARD_SIZE; ++col) {
        printf("%2d ", col + 1);
    }
    printf("\n");

    for (int row = 0; row < BOARD_SIZE; ++row) {
        printf("%2d  ", row + 1);
        for (int col = 0; col < BOARD_SIZE; ++col) {
            bool is_last = (row == last_move.row && col == last_move.col);
            printf(" %c ", cell_symbol(board->cells[row][col], is_last));
        }
        printf("\n");
    }
    printf("\n");
}

void ui_print_board(const Board *board) {
    Move none = {-1, -1};
    ui_print_board_with_last(board, none);
}

void ui_prompt_mode(void) {
    printf("Gomoku AI (15x15, Black forbidden rules enabled)\n");
    printf("Select mode:\n");
    printf("  1. Human (Black X) vs AI (White O)\n");
    printf("  2. AI (Black X) vs Human (White O)\n");
    printf("  3. Human vs Human\n");
    printf("  4. AI vs AI demo\n");
    printf("Enter choice (1-4): ");
    fflush(stdout);
}

bool ui_read_mode(GameMode *mode) {
    char line[64];
    if (fgets(line, sizeof(line), stdin) == NULL) {
        return false;
    }
    int choice = 0;
    char extra;
    if (sscanf(line, "%d %c", &choice, &extra) != 1
        || choice < 1
        || choice > 4) {
        return false;
    }
    *mode = (GameMode)choice;
    return true;
}

void ui_prompt_move(Cell player) {
    printf(
        "%s to move. Enter row and column (1-%d), or q to quit: ",
        player == CELL_BLACK ? "Black (X)" : "White (O)",
        BOARD_SIZE
    );
    fflush(stdout);
}

InputResult ui_read_move(int *row, int *col) {
    char line[128];

    if (fgets(line, sizeof(line), stdin) == NULL) {
        printf("\n");
        return INPUT_EOF;
    }

    char command[16];
    if (sscanf(line, "%15s", command) == 1
        && (strcmp(command, "q") == 0 || strcmp(command, "quit") == 0)) {
        return INPUT_QUIT;
    }

    int user_row;
    int user_col;
    char extra;

    if (sscanf(line, "%d %d %c", &user_row, &user_col, &extra) != 2
        || user_row < 1
        || user_row > BOARD_SIZE
        || user_col < 1
        || user_col > BOARD_SIZE) {
        printf("Invalid input. Enter two integers from 1 to %d.\n", BOARD_SIZE);
        return INPUT_INVALID;
    }

    *row = user_row - 1;
    *col = user_col - 1;
    return INPUT_OK;
}

void ui_print_ai_move(Cell player, Move move, const SearchStats *stats) {
    printf(
        "AI %s plays (%d, %d) | depth=%d nodes=%llu vcf=%llu time=%.3fs score=%d%s%s%s%s\n",
        player == CELL_BLACK ? "Black (X)" : "White (O)",
        move.row + 1,
        move.col + 1,
        stats->completed_depth,
        (unsigned long long)stats->nodes,
        (unsigned long long)stats->vcf_nodes,
        stats->elapsed_sec,
        stats->score,
        stats->used_vcf ? " [VCF]" : "",
        stats->used_guardrail ? " [GUARD]" : "",
        stats->used_nnue ? " [NNUE]" : "",
        stats->used_cnn ? " [CNN]" : ""
    );
}

void ui_print_move_error(MoveOutcome outcome) {
    switch (outcome.result) {
        case MOVE_INVALID:
            printf("That position is outside the board or already occupied.\n");
            break;
        case MOVE_BLACK_FORBIDDEN:
            printf(
                "Black made a forbidden move: %s. White wins.\n",
                rules_forbidden_reason_name(outcome.forbidden)
            );
            break;
        default:
            break;
    }
}

void ui_print_game_result(MoveOutcome outcome) {
    switch (outcome.result) {
        case MOVE_BLACK_WIN:
            printf("Black (X) wins with exactly five stones in a row.\n");
            break;
        case MOVE_WHITE_WIN:
            printf("White (O) wins with five or more stones in a row.\n");
            break;
        case MOVE_BLACK_FORBIDDEN:
            printf(
                "Black loses by forbidden move: %s.\n",
                rules_forbidden_reason_name(outcome.forbidden)
            );
            break;
        case MOVE_DRAW:
            printf("The board is full. The game is a draw.\n");
            break;
        default:
            break;
    }
}
