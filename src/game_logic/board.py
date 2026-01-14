from typing import List, Tuple

# Eliminamos la dependencia de numpy para el procesamiento en tiempo real
# import numpy as np  <-- Ya no lo necesitamos aquí
from src.config import BOARD_COLS, BOARD_ROWS


class Board:
    def __init__(self):
        # Inicializamos con listas de listas
        self.board = [[0 for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.reset()

    def reset(self):
        """Reinicia el tablero a su estado inicial."""
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                self.board[row][col] = 0
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

    def undo_move(self, row, col, prev_turn, prev_winner, prev_game_over, prev_win_info):
        """Revierte el tablero a un estado anterior exacto."""
        self.board[row][col] = 0
        self.turn = prev_turn
        self.winner = prev_winner
        self.game_over = prev_game_over
        self.win_info = prev_win_info

    def switch_turn(self):
        """Cambia el turno del jugador."""
        self.turn = 2 if self.turn == 1 else 1

    def is_full(self):
        """Verifica si el tablero está lleno."""
        # Forma rápida en listas de listas:
        for row in self.board:
            if 0 in row:
                return False
        return True

    def check_win(self):
        """Versión optimizada para 3x3 con listas."""
        b = self.board
        p = self.turn

        # Filas y Columnas
        for i in range(3):
            if b[i][0] == b[i][1] == b[i][2] == p:
                return True
            if b[0][i] == b[1][i] == b[2][i] == p:
                return True

        # Diagonales
        if b[0][0] == b[1][1] == b[2][2] == p:
            return True
        if b[0][2] == b[1][1] == b[2][0] == p:
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
