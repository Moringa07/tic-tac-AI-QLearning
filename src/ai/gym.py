import random

from src.ai.minimax import find_best_move_and_viz
from src.ai.qlearning import QLearningAgent
from src.game_logic.board import Board


def train(episodes=20000, minimax_ratio=0.2):
    board = Board()
    agent = QLearningAgent()

    for episode in range(episodes):
        board.reset()

        is_playing_minimax = random.random() < minimax_ratio

        agent.epsilon = max(0.01, 1.0 - (episode / episodes))

        history = {1: None, 2: None}

        while not board.game_over:
            current_player = board.turn
            state = agent.get_state_key(board.board)

            if current_player == 1:
                action = agent.choose_action(board)
            else:
                if is_playing_minimax:
                    action, _ = find_best_move_and_viz(board, use_alpha_beta=True)
                else:
                    action = agent.choose_action(board)

            if current_player == 1 or (current_player == 2 and not is_playing_minimax):
                history[current_player] = (state, action)

            board.make_move(action[0], action[1])

            if board.game_over:
                if board.winner == 1:
                    s, a = history[1]
                    agent.learn(s, a, 1, None, [], True)

                    if not (current_player == 2 and is_playing_minimax):
                        s_p, a_p = history[2]
                        agent.learn(s_p, a_p, -1, state, [], True)
                elif board.winner == 2:
                    s, a = history[1]
                    agent.learn(s, a, -1, state, [], True)

                    if not is_playing_minimax:
                        s_p, a_p = history[2]
                        agent.learn(s_p, a_p, 1, None, [], True)

                else:
                    for p in [1, 2]:
                        if history[p] is not None and not (p == 2 and is_playing_minimax):
                            s, a = history[p]
                            agent.learn(s, a, 0.5, None, [], True)

            else:
                other_player = 2 if current_player == 1 else 1
                if history[other_player] is not None and not (other_player == 2 and is_playing_minimax):
                    prev_state, prev_action = history[other_player]
                    current_board_state = agent.get_state_key(board.board)
                    agent.learn(prev_state, prev_action, 0, current_board_state, board.get_available_moves(), False)

    return agent
