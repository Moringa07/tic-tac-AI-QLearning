from typing import List, Tuple

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
        """Versión compatible con el Renderer (win_type, index)."""
        b = self.board
        p = self.turn

        # Filas
        for r in range(3):
            if b[r][0] == b[r][1] == b[r][2] == p:
                self.win_info = ("row", r)
                return True

        # Columnas
        for c in range(3):
            if b[0][c] == b[1][c] == b[2][c] == p:
                self.win_info = ("col", c)
                return True

        # Diagonal Principal
        if b[0][0] == b[1][1] == b[2][2] == p:
            self.win_info = ("diag", 1)
            return True

        # Diagonal Secundaria
        if b[0][2] == b[1][1] == b[2][0] == p:
            self.win_info = ("diag", 2)
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
