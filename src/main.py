import sys
import time
from enum import Enum

import pygame

from src.ai.minimax import (
    find_best_move_and_viz,
)
from src.config import *
from src.game_logic.board import Board
from src.gui.renderer import Renderer


# --- Enums y Constantes ---
class GameState(Enum):
    MENU = 0
    AI_SELECTION = 1
    PLAYING = 2


class PlayerType:
    HUMAN = "HUMAN"
    AI_SLOW = "AI_SLOW"
    AI_FAST = "AI_FAST"


class GameController:
    def __init__(self):
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Comparador de IA: Minimax vs Alfa-Beta")
        self.clock = pygame.time.Clock()

        self.board = Board()
        self.renderer = Renderer(self.screen)

        self.state = GameState.MENU
        self.running = True

        self.player_types = [PlayerType.HUMAN, PlayerType.HUMAN]
        self.ai_speed_selected = None
        self.last_graph_data = []
        self.waiting_for_step = False

        # Configuración de Menús
        self.menu_options = [
            "Humano vs Humano",
            "Humano vs IA (Lenta - Minimax)",
            "Humano vs IA (Rápida - AlfaBeta)",
            "IA Lenta vs IA Rápida",
            "Humano vs QLearning",
        ]
        self.menu_selection = 0

        self.start_options = ["Humano Inicia", "IA Inicia"]
        self.start_selection = 0

        self.menu_rects = []

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

    def handle_events(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
                return

            if self.state == GameState.MENU:
                self._handle_menu_input(event, self.menu_options, is_main_menu=True)

            elif self.state == GameState.AI_SELECTION:
                self._handle_menu_input(event, self.start_options, is_main_menu=False)

            elif self.state == GameState.PLAYING:
                self._handle_playing_input(event)

    def _handle_menu_input(self, event, options_list, is_main_menu):
        selection_attr = "menu_selection" if is_main_menu else "start_selection"
        current_selection = getattr(self, selection_attr)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                setattr(
                    self,
                    selection_attr,
                    (current_selection - 1) % len(options_list),
                )
            elif event.key == pygame.K_DOWN:
                setattr(
                    self,
                    selection_attr,
                    (current_selection + 1) % len(options_list),
                )
            elif event.key == pygame.K_RETURN:
                if is_main_menu:
                    self._confirm_main_menu_selection()
                else:
                    self._confirm_ai_selection()
            elif event.key == pygame.K_ESCAPE and not is_main_menu:
                self.state = GameState.MENU

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mouse_pos = pygame.mouse.get_pos()
            for i, rect in enumerate(self.menu_rects):
                if rect.collidepoint(mouse_pos):
                    setattr(self, selection_attr, i)
                    if is_main_menu:
                        self._confirm_main_menu_selection()
                    else:
                        self._confirm_ai_selection()
                    break

        mouse_pos = pygame.mouse.get_pos()
        for i, rect in enumerate(self.menu_rects):
            if rect.collidepoint(mouse_pos):
                setattr(self, selection_attr, i)

    def _confirm_main_menu_selection(self):
        sel = self.menu_selection
        if sel == 0:  # H v H
            self.start_game([PlayerType.HUMAN, PlayerType.HUMAN])
        elif sel == 1:  # IA Lenta
            self.ai_speed_selected = PlayerType.AI_SLOW
            self.state = GameState.AI_SELECTION
        elif sel == 2:  # IA Rápida
            self.ai_speed_selected = PlayerType.AI_FAST
            self.state = GameState.AI_SELECTION
        elif sel == 3:  # IA v IA
            self.start_game([PlayerType.AI_FAST, PlayerType.AI_SLOW])

    def _confirm_ai_selection(self):
        p1 = PlayerType.HUMAN
        p2 = self.ai_speed_selected

        if self.start_selection == 0:
            self.start_game([p1, p2])
        else:
            self.start_game([p2, p1])

    def _handle_playing_input(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                self.reset_board()
            elif event.key == pygame.K_ESCAPE:
                self.state = GameState.MENU
            elif event.key == pygame.K_RETURN and self.waiting_for_step:
                self.waiting_for_step = False

        if event.type == pygame.MOUSEBUTTONDOWN and not self.board.game_over:
            current_player = self.player_types[self.board.turn - 1]
            if current_player == PlayerType.HUMAN:
                self._process_human_click(event.pos)

    def _process_human_click(self, pos):
        mouseX, mouseY = pos
        if (
            self.renderer.board_offset_x <= mouseX < self.renderer.board_offset_x + BOARD_WIDTH
            and mouseY > BOARD_OFFSET_Y
        ):
            clicked_row = (mouseY - BOARD_OFFSET_Y) // SQUARE_SIZE
            clicked_col = (mouseX - self.renderer.board_offset_x) // SQUARE_SIZE

            if 0 <= clicked_row < BOARD_ROWS and 0 <= clicked_col < BOARD_COLS:
                self.board.make_move(clicked_row, clicked_col)

    def update(self):
        if self.state == GameState.PLAYING:
            self._update_game_logic()

    def _update_game_logic(self):
        if self.waiting_for_step:
            return

        current_player = self.player_types[self.board.turn - 1]

        if not self.board.game_over and current_player in [
            PlayerType.AI_SLOW,
            PlayerType.AI_FAST,
        ]:
            self._execute_ai_turn(current_player)

    def _execute_ai_turn(self, ai_type):
        self.draw()
        start_time = time.time()

        print(f"Turno {self.board.turn} ({ai_type}): Pensando...")

        use_alpha_beta = ai_type == PlayerType.AI_FAST

        move, tree_data = find_best_move_and_viz(self.board, use_alpha_beta=use_alpha_beta)

        self.last_graph_data = tree_data

        print(f"Cálculo completado en: {time.time() - start_time:.4f}s")

        if move:
            self.board.make_move(move[0], move[1])

            is_p1_ai = self.player_types[0] in [
                PlayerType.AI_SLOW,
                PlayerType.AI_FAST,
            ]
            is_p2_ai = self.player_types[1] in [
                PlayerType.AI_SLOW,
                PlayerType.AI_FAST,
            ]

            if is_p1_ai and is_p2_ai:
                self.waiting_for_step = True

    def draw(self):
        if self.state == GameState.MENU:
            self.menu_rects = self.renderer.draw_menu(self.menu_options, self.menu_selection)

        elif self.state == GameState.AI_SELECTION:
            self.menu_rects = self.renderer.draw_menu(self.start_options, self.start_selection)

        elif self.state == GameState.PLAYING:
            self._draw_game_screen()

        pygame.display.update()

    def _draw_game_screen(self):
        is_h_vs_h = self.player_types == [PlayerType.HUMAN, PlayerType.HUMAN]
        self.renderer.set_centered(is_h_vs_h)

        should_invert = (
            len(self.player_types) == 2
            and self.player_types[1] == PlayerType.HUMAN
            and self.player_types[0] != PlayerType.HUMAN
        )
        self.renderer.set_inverted(should_invert)

        self.renderer.draw_grid()
        self.renderer.draw_symbols(self.board.board)
        self.renderer.draw_decision_graph(self.last_graph_data)

        real_current_player = self.player_types[self.board.turn - 1]
        label = "HUMAN" if real_current_player == PlayerType.HUMAN else "IA"
        self.renderer.draw_turn_indicator(self.board.turn, label)

        current_player = self.player_types[self.board.turn - 1]
        if not self.board.game_over and current_player == PlayerType.HUMAN:
            self._draw_ghost_symbol()

        if self.board.game_over:
            self.renderer.draw_win_line(self.board)
            self.renderer.draw_game_over_text(self.board)

        if self.waiting_for_step:
            self._draw_step_prompt()

    def _draw_ghost_symbol(self):
        mouse_pos = pygame.mouse.get_pos()
        mouseX, mouseY = mouse_pos
        if (
            self.renderer.board_offset_x <= mouseX < self.renderer.board_offset_x + BOARD_WIDTH
            and mouseY > BOARD_OFFSET_Y
        ):
            row = (mouseY - BOARD_OFFSET_Y) // SQUARE_SIZE
            col = (mouseX - self.renderer.board_offset_x) // SQUARE_SIZE
            if 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS:
                if self.board.is_valid_move(row, col):
                    self.renderer.draw_ghost_symbol(row, col, self.board.turn)

    def _draw_step_prompt(self):
        font = pygame.font.Font(None, 30)
        prompt = font.render("Presione ENTER para siguiente jugada...", True, CIRCLE_COLOR)
        rect = prompt.get_rect(center=(WIDTH // 3.25, HEIGHT - 25))
        bg = rect.inflate(20, 10)
        pygame.draw.rect(self.screen, BG_COLOR, bg)
        self.screen.blit(prompt, rect)

    def start_game(self, players):
        self.player_types = players
        self.state = GameState.PLAYING
        self.reset_board()

    def reset_board(self):
        self.board = Board()
        self.last_graph_data = []
        self.waiting_for_step = False
        pygame.event.clear()


if __name__ == "__main__":
    game = GameController()
    game.run()
