import os
import pickle
import random

from src.benchmarks.utils import evaluate_vs_minimax
from src.game_logic.board import Board


def train_with_decay(
    agent,
    episodes=20000,
    minimax_ratio=0.2,
    pickle_path="tictactoe_lookup.pkl",
    epsilon_decay_gen=None,
    reward_draw_gen=0.5,
    start_epsilon=1.0,
    progress_callback=None,
):
    """
    Entrena un agente QLearning usando la tabla precomputada como oponente maestro
    """
    if not os.path.exists(pickle_path):
        print(f"Error: No se encontró {pickle_path}. Usando Minimax real como fallback.")
        lookup_table = None
    else:
        with open(pickle_path, "rb") as f:
            lookup_table = pickle.load(f)
        # print("Tabla cargada. Iniciando entrenamiento acelerado...")

    board = Board()
    agent.epsilon = start_epsilon
    episodes_to_optimal = episodes
    optimal_threshold = 50 * 0.98

    if epsilon_decay_gen is None:
        decay_factor = start_epsilon / episodes
    else:
        decay_factor = epsilon_decay_gen

    REWARD_DRAW = reward_draw_gen

    for episode in range(episodes):
        if progress_callback:
            progress_callback(episode, episodes)

        board.reset()

        is_playing_master = random.random() < minimax_ratio

        agent.epsilon = max(0.01, agent.epsilon - decay_factor)

        history = {1: None, 2: None}

        while not board.game_over:
            current_player = board.turn
            state_key = agent.get_state_key(board.board)

            if current_player == 1:
                action = agent.choose_action(board)
            else:
                if is_playing_master and lookup_table is not None:
                    state_hash = tuple(item for row in board.board for item in row)
                    move_data = lookup_table.get(state_hash)

                    if isinstance(move_data, dict):
                        action = move_data["move"]
                    else:
                        action = move_data

                    if action is None:
                        action = random.choice(board.get_available_moves())
                else:
                    action = agent.choose_action(board)

            if current_player == 1 or (current_player == 2 and not is_playing_master):
                history[current_player] = (state_key, action)

            board.make_move(action[0], action[1])

            if board.game_over:
                if board.winner == 1:
                    s, a = history[1]
                    agent.learn(s, a, 1, None, [], True)
                    if not is_playing_master and history[2]:
                        s_p, a_p = history[2]
                        agent.learn(s_p, a_p, -1, state_key, [], True)

                elif board.winner == 2:
                    if history[1]:
                        s, a = history[1]
                        agent.learn(s, a, -1, state_key, [], True)
                    if not is_playing_master and history[2]:
                        s_p, a_p = history[2]
                        agent.learn(s_p, a_p, 1, None, [], True)

                else:
                    for p in [1, 2]:
                        if history[p] and not (p == 2 and is_playing_master):
                            s, a = history[p]
                            agent.learn(s, a, REWARD_DRAW, None, [], True)
            else:
                other_player = 2 if current_player == 1 else 1
                if history[other_player] and not (other_player == 2 and is_playing_master):
                    prev_state, prev_action = history[other_player]
                    current_board_state = agent.get_state_key(board.board)
                    agent.learn(prev_state, prev_action, 0, current_board_state, board.get_available_moves(), False)

        if episode % 100 == 0:
            wins, _, draws = evaluate_vs_minimax(agent, num_games=10)
            if (wins + draws) / 10.0 >= 1.0:
                episodes_to_optimal = episode
                break

    agent.epsilon = 0.01
    return agent, episodes_to_optimal
