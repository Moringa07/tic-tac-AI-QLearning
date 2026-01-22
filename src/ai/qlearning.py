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

        # Exploración: movimiento aleatorio
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        # Explotación: elegir la acción con el mejor valor Q
        q_values = [self.get_q_value(state, move) for move in available_moves]
        max_q = max(q_values)

        # Si hay varios movimientos con el mismo valor máximo, elegir uno al azar entre ellos
        best_moves = [move for move, q in zip(available_moves, q_values) if q == max_q]
        return random.choice(best_moves)

    def learn(self, state, action, reward, next_state, next_available_moves, done):
        """Actualiza la tabla Q usando la fórmula de Bellman."""
        old_q = self.get_q_value(state, action)

        if done:
            target = reward
        else:
            # Predicción de la mejor recompensa futura
            if next_available_moves:
                next_max_q = max([self.get_q_value(next_state, m) for m in next_available_moves])
            else:
                next_max_q = 0
            target = reward + self.gamma * next_max_q

        # Fórmula de actualización: Q(s,a) = Q(s,a) + alpha * (recompensa + gamma * maxQ(s',a') - Q(s,a))
        self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)

    def save_model(self, filename="q_table.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename="q_table.pkl"):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)
