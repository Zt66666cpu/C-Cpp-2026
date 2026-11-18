#include "board.h"
#include "console_ui.h"
#include "rules.h"

#include <stdio.h>

static Cell other_player(Cell player) {
    return player == CELL_BLACK ? CELL_WHITE : CELL_BLACK;
}

int main(void) {
    Board board;
    board_init(&board);

    Cell current_player = CELL_BLACK;

    printf("Gomoku: Black moves first.\n");

    for (;;) {
        ui_print_board(&board);
        ui_prompt_move(current_player);

        int row;
        int col;
        InputResult input = ui_read_move(&row, &col);

        if (input == INPUT_QUIT) {
            printf("Game ended by player.\n");
            break;
        }

        if (input == INPUT_EOF) {
            printf("Input ended. Game stopped.\n");
            break;
        }

        if (input == INPUT_INVALID) {
            continue;
        }

        MoveOutcome outcome = rules_apply_move(
            &board,
            row,
            col,
            current_player
        );

        if (outcome.result == MOVE_INVALID) {
            ui_print_move_error(outcome);
            continue;
        }

        if (outcome.result != MOVE_OK) {
            ui_print_board(&board);
            ui_print_game_result(outcome);
            break;
        }

        current_player = other_player(current_player);
    }

    return 0;
}
