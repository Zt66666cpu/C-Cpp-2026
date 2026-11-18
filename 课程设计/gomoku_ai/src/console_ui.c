#include "console_ui.h"

#include <stdio.h>
#include <string.h>

static char cell_symbol(Cell cell) {
    switch (cell) {
        case CELL_BLACK:
            return 'X';
        case CELL_WHITE:
            return 'O';
        case CELL_EMPTY:
        default:
            return '.';
    }
}

void ui_print_board(const Board *board) {
    printf("\n    ");
    for (int col = 0; col < BOARD_SIZE; ++col) {
        printf("%2d ", col + 1);
    }
    printf("\n");

    for (int row = 0; row < BOARD_SIZE; ++row) {
        printf("%2d  ", row + 1);
        for (int col = 0; col < BOARD_SIZE; ++col) {
            printf(" %c ", cell_symbol(board->cells[row][col]));
        }
        printf("\n");
    }
    printf("\n");
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
