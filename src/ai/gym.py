from src.ai.qlearning import QLearningAgent
from src.game_logic.board import Board


def train(episodes=20000):
    board = Board()
    agent = QLearningAgent()

    for episode in range(episodes):
        board.reset()

        # Diccionarios para recordar el último estado y acción de cada jugador
        # history[1] guardará (estado_previo, accion_previa) del jugador 1
        history = {1: None, 2: None}

        while not board.game_over:
            current_player = board.turn
            state = agent.get_state_key(board.board)
            action = agent.choose_action(board)

            # Guardamos el movimiento actual en la historia antes de ejecutarlo
            history[current_player] = (state, action)

            # El jugador actual hace su movimiento
            board.make_move(action[0], action[1])

            if board.game_over:
                if board.winner != 0:
                    # EL JUGADOR ACTUAL GANÓ
                    # 1. Recompensa positiva para el ganador
                    agent.learn(state, action, 1, None, [], True)

                    # 2. Recompensa NEGATIVA para el perdedor
                    other_player = 2 if current_player == 1 else 1
                    prev_state, prev_action = history[other_player]

                    # Castigamos su último movimiento con -1
                    agent.learn(prev_state, prev_action, -1, state, [], True)
                else:
                    # EMPATE: Recompensa moderada para ambos
                    for p in [1, 2]:
                        s, a = history[p]
                        agent.learn(s, a, 0.5, None, [], True)
            else:
                # El juego continúa.
                # Si el jugador anterior ya hizo un movimiento, actualizamos su Q-Value
                # con recompensa 0 (aún no sabemos si ese movimiento fue bueno o malo)
                other_player = 2 if current_player == 1 else 1
                if history[other_player] is not None:
                    prev_state, prev_action = history[other_player]
                    # El "next_state" para el otro jugador es el tablero después
                    # de que el jugador actual movió.
                    current_board_state = agent.get_state_key(board.board)
                    agent.learn(prev_state, prev_action, 0, current_board_state, board.get_available_moves(), False)

    return agent
