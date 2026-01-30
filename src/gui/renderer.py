import pygame

from src.config import *
from src.gui.components.board_component import BoardComponent
from src.gui.components.config_component import ConfigComponent
from src.gui.components.graph_component import GraphComponent
from src.gui.components.hud_component import HUDComponent
from src.gui.components.menu_component import MenuComponent



class Renderer:
    def __init__(self, screen):
        self.screen = screen

        self.font = pygame.font.Font(None, 45)
        self.title_font = pygame.font.Font(None, 80)

        # Tablero Normal (Centrado)
        board_rect = pygame.Rect(0, 0, BOARD_WIDTH, BOARD_WIDTH)
        board_rect.center = (WIDTH // 2, HEIGHT // 2)
        
        # Tableros Divididos (Izquierda y Derecha)
        left_rect = pygame.Rect(0, 0, SPLIT_BOARD_WIDTH, SPLIT_BOARD_WIDTH)
        left_rect.center = (WIDTH * 0.28, HEIGHT // 2 + 30) # Un poco más abajo para título
        
        right_rect = pygame.Rect(0, 0, SPLIT_BOARD_WIDTH, SPLIT_BOARD_WIDTH)
        right_rect.center = (WIDTH * 0.72, HEIGHT // 2 + 30)

        graph_rect = pygame.Rect(WIDTH * 0.55, 0, WIDTH * 0.45, HEIGHT)

        self.board_view = BoardComponent(board_rect)
        self.left_board = BoardComponent(left_rect)
        self.right_board = BoardComponent(right_rect)
        
        self.menu_view = MenuComponent(screen.get_rect(), self.font, self.title_font)
        self.config_view = ConfigComponent(screen.get_rect(), self.font)
        self.graph_view = GraphComponent(graph_rect)
        self.hud_view = HUDComponent(self.font)

        self.inverted_symbols = False
        
        # Referencia al tablero activo actual (para clicks)
        self.active_board = self.board_view 

    def set_inverted(self, inverted: bool):
        """Define si el jugador 1 es X u O."""
        self.inverted_symbols = inverted

    def set_centered(self, is_centered: bool):
        """Ajusta la posición del tablero. Ahora siempre centrado según petición."""
        # Se ignora el parámetro is_centered por petición de diseño
        self.board_view.rect.center = (WIDTH // 2, HEIGHT // 2)

        self.board_offset_x = self.board_view.rect.x
        self.board_offset_y = self.board_view.rect.y

    def draw_menu(self, options, selection):
        self.screen.fill(BG_COLOR)
        return self.menu_view.draw(self.screen, options, selection)

    def draw_custom_setup(self):
        self.screen.fill(BG_COLOR)
        self.config_view.draw(self.screen)

    def draw_training_progress(self, current, total):
        self.screen.fill(BG_COLOR)
        
        # Centro
        cx, cy = WIDTH // 2, HEIGHT // 2
        
        # Texto
        text = self.font.render("Entrenando Agente...", True, FONT_COLOR)
        text_rect = text.get_rect(center=(cx, cy - 50))
        self.screen.blit(text, text_rect)
        
        # Barra de progreso (Fondo)
        bar_width = 500
        bar_height = 30
        bar_rect = pygame.Rect(0, 0, bar_width, bar_height)
        bar_rect.center = (cx, cy + 20)
        pygame.draw.rect(self.screen, MENU_BUTTON_COLOR, bar_rect, border_radius=15)
        
        # Barra de progreso (Lleno)
        progress = current / total
        fill_width = int(bar_width * progress)
        fill_rect = pygame.Rect(bar_rect.left, bar_rect.top, fill_width, bar_height)
        pygame.draw.rect(self.screen, MENU_SELECTED_COLOR, fill_rect, border_radius=15)
        
        # Porcentaje
        pct_text = self.font.render(f"{int(progress * 100)}%", True, FONT_COLOR)
        self.screen.blit(pct_text, pct_text.get_rect(center=(cx, cy + 70)))
        
        pygame.display.flip()

    def draw_grid(self):
        self.screen.fill(BG_COLOR)
        self.active_board.draw_grid(self.screen)

    def draw_symbols(self, board_array):
        # Este método se usa poco directamente, pero por compatibilidad:
        self.active_board.draw_symbols(self.screen, board_array, self.inverted_symbols)

    def draw_decision_graph(self, tree_data):
        self.graph_view.draw(self.screen, tree_data, self.inverted_symbols)

    def draw_turn_indicator(self, turn, label):
        self.hud_view.draw_turn(self.screen, self.active_board.rect, turn, label, self.inverted_symbols)

    def draw_ghost_symbol(self, row, col, turn):
        self.active_board.draw_ghost(self.screen, row, col, turn, self.inverted_symbols)

    def draw_win_line(self, board):
        self.hud_view.draw_win_line(self.screen, self.active_board, board)

    def draw_game_over_text(self, board):
        self.hud_view.draw_game_over(self.screen, self.screen.get_rect(), board)

    def get_cell_from_mouse(self, mouse_pos):
        """Delegar la detección del mouse al tablero activo."""
        return self.active_board.get_cell_from_mouse(mouse_pos)

    def draw_game(self, board, player_types, ai_tree, q_values=None):
        """Método orquestador que dibuja toda la pantalla de juego."""
        self.screen.fill(BG_COLOR)

        if q_values is not None:
            # --- MODO DIVIDIDO (Juego + Cerebro) ---
            self.active_board = self.left_board # Clicks van al izquierdo
            
            # 1. Dibujar Tablero Jugable (Izquierda)
            self.left_board.draw_grid(self.screen)
            self.left_board.draw_symbols(self.screen, board.board, self.inverted_symbols)
            
            # Título Tablero Juego (Bajado un poco más y centrado)
            lbl_game = self.font.render("Juego", True, FONT_COLOR)
            # Centerx alineado al tablero, Top ajustado +20 px desde la posición anterior
            lbl_g_rect = lbl_game.get_rect(centerx=self.left_board.rect.centerx, bottom=self.left_board.rect.top - 15)
            self.screen.blit(lbl_game, lbl_g_rect)

            # 2. Dibujar Tablero Cerebro (Derecha)
            self.right_board.draw_grid(self.screen)
            # Pasamos q_values aquí. También los símbolos actuales semitransparentes (opcional, pero ayuda)
            self.right_board.draw_symbols(self.screen, board.board, self.inverted_symbols, q_values=q_values)
            
            # Título Tablero Cerebro
            lbl_brain = self.font.render("Evaluación IA", True, FONT_COLOR)
            lbl_b_rect = lbl_brain.get_rect(centerx=self.right_board.rect.centerx, bottom=self.right_board.rect.top - 15)
            self.screen.blit(lbl_brain, lbl_b_rect)

        else:
            # --- MODO NORMAL (Único, Centrado) ---
            self.active_board = self.board_view # Clicks van al central
            
            self.board_view.draw_grid(self.screen)
            self.board_view.draw_symbols(self.screen, board.board, self.inverted_symbols)

        if ai_tree:
            self.graph_view.draw(self.screen, ai_tree, self.inverted_symbols)

        current_player_idx = board.turn - 1
        player_label = "HUMAN" if player_types[current_player_idx] == PlayerType.HUMAN else "IA"

        self.hud_view.draw_turn(self.screen, self.screen.get_rect(), board.turn, player_label, self.inverted_symbols)

        if board.game_over:
            self.hud_view.draw_win_line(self.screen, self.left_board if q_values else self.board_view, board)
            self.hud_view.draw_game_over(self.screen, self.screen.get_rect(), board)

    def draw_step_prompt(self):
        """Interfaz para el mensaje de espera de la IA."""
        self.hud_view.draw_step_prompt(self.screen)
