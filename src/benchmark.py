import os

from src.ai.minimax import find_best_move_and_viz
from src.ai.qlearning import QLearningAgent
from src.game_logic.board import Board


def run_benchmark(num_games=50):
    ql_agent = QLearningAgent(epsilon=0)
    model_path = "src/models/q_table.pkl"

    if not os.path.exists(model_path):
        print("Error: No se encontró el modelo entrenado en src/models/q_table.pkl")
        return

    ql_agent.load_model(model_path)

    stats = {"wins": 0, "losses": 0, "draws": 0}

    print(f"Iniciando Benchmark: Q-Learning vs Minimax ({num_games} partidas)...")
    print("-" * 50)

    for i in range(num_games):
        board = Board()
        ql_is_p1 = i % 2 == 0

        while not board.game_over:
            current_player = board.turn

            is_ql_turn = (current_player == 1 and ql_is_p1) or (current_player == 2 and not ql_is_p1)

            if is_ql_turn:
                move = ql_agent.choose_action(board)
            else:
                move, _ = find_best_move_and_viz(board, use_alpha_beta=True)

            if move:
                board.make_move(move[0], move[1])

        if board.winner == 0:
            stats["draws"] += 1
            result_str = "Empate"
        elif (board.winner == 1 and ql_is_p1) or (board.winner == 2 and not ql_is_p1):
            stats["wins"] += 1
            result_str = "Victoria QL"
        else:
            stats["losses"] += 1
            result_str = "Victoria Minimax"

        print(f"Partida {i + 1:02d}: {result_str} (QL inicia: {ql_is_p1})")

    total = num_games
    win_rate = (stats["wins"] / total) * 100
    loss_rate = (stats["losses"] / total) * 100
    draw_rate = (stats["draws"] / total) * 100

    print("-" * 50)
    print("RESUMEN FINAL:")
    print(f"Victorias Q-Learning: {stats['wins']} ({win_rate:.1f}%)")
    print(f"Derrotas Q-Learning:  {stats['losses']} ({loss_rate:.1f}%)")
    print(f"Empates:              {stats['draws']} ({draw_rate:.1f}%)")
    print("-" * 50)

    if loss_rate == 0:
        print("¡Resultado Perfecto! El agente Q-Learning es invencible.")
    else:
        print("El agente aún tiene debilidades frente a Minimax.")


if __name__ == "__main__":
    run_benchmark(50)
