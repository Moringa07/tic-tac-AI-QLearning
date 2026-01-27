import math
import pickle

from src.game_logic.board import Board


def get_board_hash(board_matrix):
    """Convierte la matriz del tablero en una tupla plana (hashable)."""
    return tuple(item for row in board_matrix for item in row)


def precompute_all_states():
    lookup_table = {}

    def solve(board: Board):
        state_hash = get_board_hash(board.board)
        if state_hash in lookup_table:
            return lookup_table[state_hash]["score"]

        # 1. Casos base
        if board.winner is not None and board.winner != 0:
            return 1 if board.winner == 1 else -1
        if board.is_full():
            return 0

        is_maximizing = board.turn == 1  # Asumamos que el jugador 1 siempre quiere max
        best_score = -math.inf if is_maximizing else math.inf
        best_move = None

        available_moves = board.get_available_moves()
        for move in available_moves:
            prev_state = (board.turn, board.winner, board.game_over, board.win_info)
            board.make_move(move[0], move[1])

            score = solve(board)

            board.undo_move(move[0], move[1], *prev_state)

            if is_maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        lookup_table[state_hash] = {"move": best_move, "score": best_score}
        return best_score

    print("Generando tabla...")
    solve(Board())

    with open("tictactoe_lookup.pkl", "wb") as f:
        pickle.dump(lookup_table, f)

    print(f"Hecho. Estados guardados: {len(lookup_table)}")


if __name__ == "__main__":
    precompute_all_states()
