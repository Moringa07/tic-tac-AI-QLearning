import os
import pickle

from src.ai.minimax import find_best_move_alpha_beta
from src.ai.ql_agent import QLearningAgent
from src.game_logic.board import Board


def evaluate_vs_minimax(ql_agent: QLearningAgent, num_games=20, pickle_path="tictactoe_lookup.pkl") -> tuple:
    """
    Evalúa el rendimiento de un agente Q-Learning contra un agente Minimax perfecto.
    Usa la tabla precomputada para mayor velocidad y recurre al minimax real si es necesario.
    """
    # 1. Intentar cargar la tabla
    lookup_table = None
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            lookup_table = pickle.load(f)

    ql_agent.epsilon = 0.01
    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        board = Board()
        ql_is_p1 = i % 2 == 0

        while not board.game_over:
            current_player = board.turn
            is_ql_turn = (current_player == 1 and ql_is_p1) or (current_player == 2 and not ql_is_p1)

            if is_ql_turn:
                move = ql_agent.choose_action(board)
            else:
                state_hash = tuple(item for row in board.board for item in row)
                move_data = lookup_table.get(state_hash) if lookup_table else None

                if move_data is not None:
                    move = move_data["move"] if isinstance(move_data, dict) else move_data
                else:
                    move, _ = find_best_move_alpha_beta(board)

            if move:
                board.make_move(move[0], move[1])

        # Conteo de resultados
        if board.winner == 0 or board.winner is None:
            draws += 1
        elif (board.winner == 1 and ql_is_p1) or (board.winner == 2 and not ql_is_p1):
            wins += 1
        else:
            losses += 1

    return wins, losses, draws
