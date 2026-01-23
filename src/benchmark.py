import glob
import os

from src.ai.minimax import find_best_move_and_viz
from src.ai.qlearning import QLearningAgent
from src.game_logic.board import Board


def evaluate_model(model_path, num_games=20):
    agent = QLearningAgent(epsilon=0)
    agent.load_model(model_path)

    wins, losses, draws = 0, 0, 0

    for i in range(num_games):
        board = Board()
        ql_is_p1 = i % 2 == 0

        while not board.game_over:
            if (board.turn == 1 and ql_is_p1) or (board.turn == 2 and not ql_is_p1):
                move = agent.choose_action(board)
            else:
                move, _ = find_best_move_and_viz(board, use_alpha_beta=True)

            if move:
                board.make_move(move[0], move[1])

        if board.winner == 0:
            draws += 1
        elif (board.winner == 1 and ql_is_p1) or (board.winner == 2 and not ql_is_p1):
            wins += 1
        else:
            losses += 1

    return wins, losses, draws


def run_tournament():
    model_files = sorted(glob.glob("src/models/q_table_*.pkl"), key=os.path.getmtime)

    print(f"{'MODELO':<25} | {'W':<3} | {'L':<3} | {'D':<3} | {'EFICACIA':<10}")
    print("-" * 60)

    for path in model_files:
        name = os.path.basename(path)
        w, l, d = evaluate_model(path)

        efficiency = ((w + d) / (w + l + d)) * 100

        print(f"{name:<25} | {w:<3} | {l:<3} | {d:<3} | {efficiency:>8.1f}%")


if __name__ == "__main__":
    run_tournament()
