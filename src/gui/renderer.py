from typing import List
import math

import pygame

from src.config import *


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen: pygame.Surface = screen
        self.font: pygame.font.Font = pygame.font.Font(None, 45)
        self.menu_font: pygame.font.Font = pygame.font.Font(None, MENU_FONT_SIZE)
        
        # Dynamic Offsets (Start with default left-aligned)
        self.board_offset_x = BOARD_OFFSET_X
        self.board_offset_y = BOARD_OFFSET_Y

    def set_centered(self, is_centered: bool):
        """Toggle between centered board (for H vs H) and left-aligned."""
        if is_centered:
            self.board_offset_x = (WIDTH - BOARD_WIDTH) // 2
        else:
            self.board_offset_x = BOARD_OFFSET_X

    def draw_menu(self, options: List[str], selected_option: int) -> List[pygame.Rect]:
        """Dibuja el menú y retorna los rectángulos de las opciones para detección de clic."""
        self.screen.fill(BG_COLOR)
        
        option_rects = []

        # Título con sombra
        title_font = pygame.font.Font(None, 80)
        title_text = title_font.render("Tres en Raya", True, (0, 0, 0)) # Sombra negra
        title_rect = title_text.get_rect(center=(WIDTH // 2 + 4, HEIGHT // 4 + 4))
        self.screen.blit(title_text, title_rect)
        
        title_text_main = title_font.render("Tres en Raya", True, FONT_COLOR)
        title_rect_main = title_text_main.get_rect(center=(WIDTH // 2, HEIGHT // 4))
        self.screen.blit(title_text_main, title_rect_main)

        # Opciones del menú
        for i, option in enumerate(options):
            center_x = WIDTH // 2
            center_y = HEIGHT // 2 + i * (FONT_SIZE + 35)
            
            # Texto base
            if i == selected_option:
                color = MENU_SELECTED_COLOR
                text_content = f">  {option}  <"
            else:
                color = (180, 180, 180) # Gris claro
                text_content = option

            text = self.font.render(text_content, True, color)
            rect = text.get_rect(center=(center_x, center_y))
            
            # Dibujar fondo si está seleccionado
            if i == selected_option:
                bg_rect = rect.inflate(40, 15)
                # Efecto visual: borde redondeado y color de fondo sutil
                pygame.draw.rect(self.screen, (68, 71, 90), bg_rect, border_radius=15)
                pygame.draw.rect(self.screen, MENU_SELECTED_COLOR, bg_rect, 2, border_radius=15)

            self.screen.blit(text, rect)
            
            # Guardar el rectángulo de colisión (un poco más grande para facilitar el clic)
            hitbox = rect.inflate(40, 20)
            option_rects.append(hitbox)
            
        return option_rects

    def draw_turn_indicator(self, turn: int, player_type: str):
        """Dibuja quién está jugando en la parte superior."""
        if player_type == "HUMAN":
            text_str = f"Turno del Humano ({'X' if turn == 1 else 'O'})"
            color = (139, 233, 253) # Cyan
        else:
            text_str = f"Turno de la IA ({'X' if turn == 1 else 'O'})"
            color = (255, 121, 198) # Pink
            
        text = self.font.render(text_str, True, color)
        # Centrar en el área superior (encima del tablero)
        # Usamos self.board_offset_x para centrarlo respecto a la tabla visualmente
        center_x = self.board_offset_x + BOARD_WIDTH // 2
        text_rect = text.get_rect(center=(center_x, BOARD_OFFSET_Y // 2))
        self.screen.blit(text, text_rect)

    def draw_ghost_symbol(self, row, col, turn):
        """Dibuja un símbolo semitransparente (hover)."""
        center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2 + self.board_offset_x
        center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2 + BOARD_OFFSET_Y
        
        # Crear superficie temporal con transparencia
        ghost_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        
        alpha = 100 # Transparencia (0-255)
        
        if turn == 1: # X
            margin = SQUARE_SIZE // 4
            color = (*CROSS_COLOR, alpha)
            start_desc = (margin, margin)
            end_desc = (SQUARE_SIZE - margin, SQUARE_SIZE - margin)
            start_asc = (margin, SQUARE_SIZE - margin)
            end_asc = (SQUARE_SIZE - margin, margin)
            
            pygame.draw.line(ghost_surf, color, start_desc, end_desc, CROSS_WIDTH)
            pygame.draw.line(ghost_surf, color, start_asc, end_asc, CROSS_WIDTH)
            
        else: # O
            color = (*CIRCLE_COLOR, alpha)
            radius = CIRCLE_RADIUS
            center = (SQUARE_SIZE // 2, SQUARE_SIZE // 2)
            pygame.draw.circle(ghost_surf, color, center, radius, CIRCLE_WIDTH)

        self.screen.blit(ghost_surf, (col * SQUARE_SIZE + self.board_offset_x, row * SQUARE_SIZE + BOARD_OFFSET_Y))

    def draw_grid(self):
        """Dibuja las líneas del tablero y el borde completo."""
        self.screen.fill(BG_COLOR)
        
        # Borde completo del tablero (coincide con el divisor)
        # Rectángulo desplazado
        # Left = self.board_offset_x
        # Top = BOARD_OFFSET_Y
        # Width = BOARD_WIDTH
        # Height = BOARD_HEIGHT area (HEIGHT - OFFSETS - MARGINS?) actually just square * rows
        board_height = BOARD_ROWS * SQUARE_SIZE
        pygame.draw.rect(self.screen, SQUARE_HOVER_COLOR, (self.board_offset_x, BOARD_OFFSET_Y, BOARD_WIDTH, board_height), 3)

        # Líneas horizontales
        for i in range(1, BOARD_ROWS):
            y = i * SQUARE_SIZE + BOARD_OFFSET_Y
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                (self.board_offset_x, y),
                (BOARD_WIDTH + self.board_offset_x, y),
                LINE_WIDTH,
            )
        # Líneas verticales
        for i in range(1, BOARD_COLS):
            x = i * SQUARE_SIZE + self.board_offset_x
            pygame.draw.line(
                self.screen,
                LINE_COLOR,
                (x, BOARD_OFFSET_Y),
                (x, board_height + BOARD_OFFSET_Y),
                LINE_WIDTH,
            )

    def draw_symbols(self, board_array):
        """Dibuja las X y O en el tablero."""
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS):
                center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2 + self.board_offset_x
                center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2 + BOARD_OFFSET_Y 

                if board_array[row][col] == 1:
                    margin = SQUARE_SIZE // 4
                    start_desc = (center_x - margin, center_y - margin)
                    end_desc = (center_x + margin, center_y + margin)
                    start_asc = (center_x - margin, center_y + margin)
                    end_asc = (center_x + margin, center_y - margin)
                    pygame.draw.line(
                        self.screen, CROSS_COLOR, start_desc, end_desc, CROSS_WIDTH
                    )
                    pygame.draw.line(
                        self.screen, CROSS_COLOR, start_asc, end_asc, CROSS_WIDTH
                    )
                elif board_array[row][col] == 2:
                    pygame.draw.circle(
                        self.screen,
                        CIRCLE_COLOR,
                        (center_x, center_y),
                        CIRCLE_RADIUS,
                        CIRCLE_WIDTH,
                    )

    def draw_win_line(self, board):
        """Dibuja la línea que tacha la jugada ganadora con efecto 'Glow'."""
        if not board.win_info:
            return

        win_type, index = board.win_info

        if win_type == "row":
            y_pos = index * SQUARE_SIZE + SQUARE_SIZE // 2 + BOARD_OFFSET_Y
            start_pos = (15 + self.board_offset_x, y_pos)
            end_pos = (BOARD_WIDTH - 15 + self.board_offset_x, y_pos)
        elif win_type == "col":
            x_pos = index * SQUARE_SIZE + SQUARE_SIZE // 2 + self.board_offset_x
            start_pos = (x_pos, 15 + BOARD_OFFSET_Y)
            end_pos = (x_pos, (BOARD_ROWS * SQUARE_SIZE) - 15 + BOARD_OFFSET_Y)
        elif win_type == "diag":
            if index == 1:  # Descendente
                start_pos = (15 + self.board_offset_x, 15 + BOARD_OFFSET_Y)
                end_pos = (BOARD_WIDTH - 15 + self.board_offset_x, (BOARD_ROWS * SQUARE_SIZE) - 15 + BOARD_OFFSET_Y)
            else:  # Ascendente
                start_pos = (15 + self.board_offset_x, (BOARD_ROWS * SQUARE_SIZE) - 15 + BOARD_OFFSET_Y)
                end_pos = (BOARD_WIDTH - 15 + self.board_offset_x, 15 + BOARD_OFFSET_Y)

        # Efecto Neon Glow
        # Dibujamos varias líneas con distinta transparencia y grosor
        glow_colors = [
            (*WIN_LINE_COLOR, 50),  # Ancho, muy transparente
            (*WIN_LINE_COLOR, 100), # Medio
            (*WIN_LINE_COLOR, 255), # Centro sólido
        ]
        widths = [LINE_WIDTH + 15, LINE_WIDTH + 5, LINE_WIDTH]
        
        # Necesitamos una surface con alpha para el glow
        glow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        
        for i in range(3):
            pygame.draw.line(glow_surf, glow_colors[i], start_pos, end_pos, widths[i])
            
        self.screen.blit(glow_surf, (0, 0))

    def draw_game_over_text(self, board):
        """Muestra un mensaje al final del juego."""
        if not board.game_over:
            return

        if board.winner == 0:
            text = "¡Es un empate!"
        else:
            text = f"¡Jugador {board.winner} gana!"

        rendered_text = self.font.render(text, True, FONT_COLOR)
        text_rect = rendered_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))

        restart_text = self.font.render("Pulsa 'R' para reiniciar", True, FONT_COLOR)
        restart_rect = restart_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))

        # Fondo semi-transparente para el texto
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        self.screen.blit(rendered_text, text_rect)
        self.screen.blit(restart_text, restart_rect)

    def draw_mini_board(self, x, y, size, board_state):
        """Dibuja un mini tablero en la posición (x, y)."""
        # Fondo del mini tablero
        rect = pygame.Rect(x, y, size, size)
        pygame.draw.rect(self.screen, (255, 255, 255), rect)
        pygame.draw.rect(self.screen, LINE_COLOR, rect, 2)

        cell_size = size // 3

        # Líneas
        for i in range(1, 3):
            pygame.draw.line(self.screen, (0, 0, 0), (x, y + i * cell_size), (x + size, y + i * cell_size), 1)
            pygame.draw.line(self.screen, (0, 0, 0), (x + i * cell_size, y), (x + i * cell_size, y + size), 1)

        # Símbolos
        font_mini = pygame.font.Font(None, int(cell_size * 1.5))
        for r in range(3):
            for c in range(3):
                val = board_state[r][c]
                if val != 0:
                    symbol = "X" if val == 1 else "O"
                    color = (0, 0, 0)
                    text = font_mini.render(symbol, True, color)
                    text_rect = text.get_rect(center=(x + c * cell_size + cell_size // 2, y + r * cell_size + cell_size // 2))
                    self.screen.blit(text, text_rect)

    def draw_decision_graph(self, graph_data: List[dict]):
        """Dibuja el grafo de decisiones usando mini tableros."""
        if not graph_data:
            return

        # Área del grafo: x = BOARD_WIDTH + OFFSET_X + MARGIN a WIDTH
        graph_start_x = BOARD_WIDTH + BOARD_OFFSET_X
        area_width = WIDTH - graph_start_x
        center_x = graph_start_x + area_width // 2
        
        # Tamaño de los mini tableros
        mini_size = 50
        
        # Nodo raíz (Estado actual - Placeholder o texto)
        root_x = center_x - mini_size // 2
        root_y = 50
        
        # Etiqueta Start
        root_label = self.font.render("Start", True, TEXT_COLOR)
        root_label_rect = root_label.get_rect(center=(center_x, 30))
        self.screen.blit(root_label, root_label_rect)
        
        # Nota: No tenemos el estado del tablero raíz pasado explícitamente aquí, 
        # así que dibujaremos un círculo o caja como "Raíz" simbólica
        root_pos = (center_x, root_y + mini_size // 2)
        pygame.draw.circle(self.screen, NODE_COLOR, root_pos, 10)

        # Dibujar hijos (movimientos evaluados)
        num_children = len(graph_data)
        if num_children == 0:
            return

        # Layout Grid para los hijos
        # Intentar encajar en columnas/filas si son muchos
        cols = 3
        rows = math.ceil(num_children / cols)
        
        start_y = 150
        spacing_x = area_width // cols
        spacing_y = 100
        
        for i, node in enumerate(graph_data):
            row = i // cols
            col = i % cols
            
            child_x = graph_start_x + (col * spacing_x) + (spacing_x - mini_size) // 2
            child_y = start_y + (row * spacing_y)
            
            # Línea desde raíz
            child_center = (child_x + mini_size // 2, child_y)
            pygame.draw.line(self.screen, LINE_COLOR, root_pos, child_center, 1)
            
            # Dibujar Mini Tablero
            if 'board' in node:
                self.draw_mini_board(child_x, child_y, mini_size, node['board'])
            
            # Score
            score = node['score']
            score_color = POSITIVE_COLOR if score > 0 else (NEGATIVE_COLOR if score < 0 else NEUTRAL_COLOR)
            
            font_small = pygame.font.Font(None, 24)
            score_text = font_small.render(f"Val: {score}", True, score_color)
            score_rect = score_text.get_rect(center=(child_x + mini_size // 2, child_y + mini_size + 15))
            self.screen.blit(score_text, score_rect)
