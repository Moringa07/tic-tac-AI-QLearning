from typing import List, Tuple

import numpy as np

from src.config import BOARD_COLS, BOARD_ROWS


class Board:
    def __init__(self):
        self.board = np.zeros((BOARD_ROWS, BOARD_COLS))
        self.reset()

    def reset(self):
        """Reinicia el tablero a su estado inicial."""
        self.board.fill(0)
        self.winner = 0
        self.turn = 1
        self.game_over = False
        self.win_info = None

    def is_valid_move(self, row, col):
        """Verifica si una casilla está vacía."""
        return self.board[row][col] == 0

    def make_move(self, row, col):
        """Realiza un movimiento y actualiza el estado del juego."""
        if self.is_valid_move(row, col) and not self.game_over:
            self.board[row][col] = self.turn
            if self.check_win():
                self.winner = self.turn
                self.game_over = True
            elif self.is_full():
                self.game_over = True  # Es un empate
            else:
                self.switch_turn()
            return True
        return False

    def switch_turn(self):
        """Cambia el turno del jugador."""
        self.turn = 2 if self.turn == 1 else 1

    def is_full(self):
        """Verifica si el tablero está lleno."""
        return 0 not in self.board

    def check_win(self):
        """Verifica todas las condiciones de victoria."""
        player = self.turn
        # Comprobar filas
        for row in range(BOARD_ROWS):
            if np.all(self.board[row] == player):
                self.win_info = ("row", row)
                return True
        # Comprobar columnas
        for col in range(BOARD_COLS):
            if np.all(self.board[:, col] == player):
                self.win_info = ("col", col)
                return True
        # Comprobar diagonales
        if np.all(np.diag(self.board) == player):
            self.win_info = ("diag", 1)  # Diagonal descendente
            return True
        if np.all(np.diag(np.fliplr(self.board)) == player):
            self.win_info = ("diag", 2)  # Diagonal ascendente
            return True

        return False

    def get_available_moves(self) -> List[Tuple[int, int]]:
        """Retorna una lista de tuplas (fila, col) para las casillas vacías."""
        moves: List[Tuple[int, int]] = []
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                if self.board[row][col] == 0:
                    moves.append((row, col))
        return moves
