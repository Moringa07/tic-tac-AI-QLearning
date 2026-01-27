import pickle
import random
from typing import List, Tuple


class QLearningAgent:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # (estado, accion) -> valor_q
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento (importancia de recompensas futuras)
        self.epsilon = epsilon  # Tasa de exploración

    def get_state_key(self, board_list: List[List[int]]) -> tuple:
        """Convierte el tablero (lista de listas) en una tupla inmutable."""
        return tuple(tuple(row) for row in board_list)

    def get_q_value(self, state, action) -> float:
        """Obtiene el valor Q de la tabla, retorna 0 si no existe."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, board) -> Tuple[int, int]:
        """Elige una acción usando la política epsilon-greedy."""
        available_moves = board.get_available_moves()
        state = self.get_state_key(board.board)

        if random.random() < self.epsilon:
            return random.choice(available_moves)

        q_values = [self.get_q_value(state, move) for move in available_moves]
        max_q = max(q_values)

        best_moves = [move for move, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_moves)

    def learn(self, state, action, reward, next_state, next_available_moves, done):
        """Actualiza la tabla Q usando la fórmula de Bellman."""
        old_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            if next_available_moves:
                next_max_q = max([self.get_q_value(next_state, m) for m in next_available_moves])
            else:
                next_max_q = 0
            target = reward + self.gamma * next_max_q

        # Fórmula de actualización: Q(s,a) = Q(s,a) + alpha * (recompensa + gamma * maxQ(s',a') - Q(s,a))
        self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)

    def save_model(self, filename="src/models/q_table.pkl"):
        import os

        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)


class GeneticQLAgent:
    def __init__(self, agent_id, gen=None):
        self.id = agent_id
        self.agent = None  # Instancia de QLearningAgent
        self.fitness = 0.0

        # El Cromosoma (α, γ, epsilon_final, N_episodes)
        if gen is None:
            self.alpha = round(random.uniform(0.01, 0.99), 2)
            self.gamma = round(random.uniform(0.01, 0.99), 2)
            self.epsilon_decay_rate = round(random.uniform(0.0001, 0.01), 4)
        else:
            self.alpha, self.gamma, self.epsilon_decay_rate = gen

    def instantiate_agent(self):
        self.agent = QLearningAgent(alpha=self.alpha, gamma=self.gamma, epsilon=1.0)
