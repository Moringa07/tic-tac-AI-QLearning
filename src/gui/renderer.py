import pygame

from src.config import *
from src.gui.components.board_component import BoardComponent
from src.gui.components.graph_component import GraphComponent
from src.gui.components.hud_component import HUDComponent
from src.gui.components.menu_component import MenuComponent


class Renderer:
    def __init__(self, screen):
        self.screen = screen

        self.font = pygame.font.Font(None, 45)
        self.title_font = pygame.font.Font(None, 80)

        board_rect = pygame.Rect(0, 0, 400, 400)
        board_rect.center = (WIDTH // 2, HEIGHT // 2)

        graph_rect = pygame.Rect(WIDTH * 0.55, 0, WIDTH * 0.45, HEIGHT)

        self.board_view = BoardComponent(board_rect)
        self.menu_view = MenuComponent(screen.get_rect(), self.font, self.title_font)
        self.graph_view = GraphComponent(graph_rect)
        self.hud_view = HUDComponent(self.font)

        self.inverted_symbols = False

    def set_inverted(self, inverted: bool):
        """Define si el jugador 1 es X u O."""
        self.inverted_symbols = inverted

    def set_centered(self, is_centered: bool):
        """Ajusta la posición del tablero según el modo de juego."""
        if is_centered:
            self.board_view.rect.center = (WIDTH // 2, HEIGHT // 2)
        else:
            self.board_view.rect.x = 50
            self.board_view.rect.centery = HEIGHT // 2

        self.board_offset_x = self.board_view.rect.x
        self.board_offset_y = self.board_view.rect.y

    def draw_menu(self, options, selection):
        self.screen.fill(BG_COLOR)
        return self.menu_view.draw(self.screen, options, selection)

    def draw_grid(self):
        self.screen.fill(BG_COLOR)
        self.board_view.draw_grid(self.screen)

    def draw_symbols(self, board_array):
        self.board_view.draw_symbols(self.screen, board_array, self.inverted_symbols)

    def draw_decision_graph(self, tree_data):
        self.graph_view.draw(self.screen, tree_data, self.inverted_symbols)

    def draw_turn_indicator(self, turn, label):
        self.hud_view.draw_turn(self.screen, self.board_view.rect, turn, label, self.inverted_symbols)

    def draw_ghost_symbol(self, row, col, turn):
        self.board_view.draw_ghost(self.screen, row, col, turn, self.inverted_symbols)

    def draw_win_line(self, board):
        self.hud_view.draw_win_line(self.screen, self.board_view, board)

    def draw_game_over_text(self, board):
        self.hud_view.draw_game_over(self.screen, self.screen.get_rect(), board)

    def draw_game(self, board, player_types, ai_tree):
        """Método orquestador que dibuja toda la pantalla de juego."""
        self.screen.fill(BG_COLOR)

        self.board_view.draw_grid(self.screen)
        self.board_view.draw_symbols(self.screen, board.board, self.inverted_symbols)

        if ai_tree:
            self.graph_view.draw(self.screen, ai_tree, self.inverted_symbols)

        current_player_idx = board.turn - 1

        player_label = "HUMAN" if player_types[current_player_idx] == PlayerType.HUMAN else "IA"

        self.hud_view.draw_turn(self.screen, self.board_view.rect, board.turn, player_label, self.inverted_symbols)

        if board.game_over:
            self.hud_view.draw_win_line(self.screen, self.board_view, board)
            self.hud_view.draw_game_over(self.screen, self.screen.get_rect(), board)

    def draw_step_prompt(self):
        """Interfaz para el mensaje de espera de la IA."""
        self.hud_view.draw_step_prompt(self.screen)
