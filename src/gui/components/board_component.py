import pygame

from src.config import CIRCLE_COLOR, CROSS_COLOR, LINE_COLOR, LINE_WIDTH
from src.gui.shapes import ShapeDrawer


class BoardComponent:
    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.rows = 3
        self.cols = 3

        self.cell_w = self.rect.width // self.cols
        self.cell_h = self.rect.height // self.rows

    def draw_grid(self, surface):
        pygame.draw.rect(surface, LINE_COLOR, self.rect, 3)

        # Líneas verticales
        for i in range(1, self.cols):
            x = self.rect.left + i * self.cell_w
            pygame.draw.line(surface, LINE_COLOR, (x, self.rect.top), (x, self.rect.bottom), LINE_WIDTH)

        # Líneas horizontales
        for i in range(1, self.rows):
            y = self.rect.top + i * self.cell_h
            pygame.draw.line(surface, LINE_COLOR, (self.rect.left, y), (self.rect.right, y), LINE_WIDTH)

    def draw_symbols(self, surface, board_array, inverted, q_values=None):
        """
        Dibuja los símbolos y opcionalmente los valores Q.
        q_values: Diccionario {(row, col): q_value} o similar.
        """
        # Dibujar Q-Values primero (al fondo)
        if q_values:
            font_small = pygame.font.Font(None, 24)
            for (r, c), q_val in q_values.items():
                if board_array[r][c] == 0: # Solo en celdas vacías
                    cell_rect = pygame.Rect(
                        self.rect.left + c * self.cell_w, self.rect.top + r * self.cell_h, self.cell_w, self.cell_h
                    )
                    
                    # Mapa de calor mejorado (Premio/Castigo)
                    # Usamos base alpha mas alta para que se vea claro
                    # Q range approx -1 to 1 (pero puede variar), saturamos en 0.5
                    
                    val_clamped = max(-1.0, min(1.0, q_val))
                    # Aumentamos el multiplicador para que valores pequeños (0.1, 0.2) se vean fuertes
                    intensity = int(min(255, abs(val_clamped * 400) + 50)) # Minimo 50 de intensidad
                    
                    if val_clamped > 0.001: # Premio (Verde) - Umbral bajo
                        bg_color = (0, 200, 0, min(180, intensity)) 
                        border_col = (0, 255, 0)
                    elif val_clamped < -0.001: # Castigo (Rojo) - Umbral bajo
                        bg_color = (200, 0, 0, min(180, intensity))
                        border_col = (255, 50, 50)
                    else: # Neutro
                        bg_color = (150, 150, 200, 50)
                        border_col = None
                    
                    # Fondo
                    overlay = pygame.Surface((self.cell_w, self.cell_h), pygame.SRCALPHA)
                    overlay.fill(bg_color)
                    surface.blit(overlay, cell_rect.topleft)
                    
                    # Borde indicador
                    if border_col:
                        pygame.draw.rect(surface, border_col, cell_rect, 2)
                    
                    # Texto del valor (Con sombra para contraste)
                    txt_str = f"{q_val:.2f}"
                    txt_color = (255, 255, 255)
                    
                    txt = font_small.render(txt_str, True, txt_color)
                    txt_rect = txt.get_rect(center=cell_rect.center)
                    
                    # Sombra negra simple
                    shadow = font_small.render(txt_str, True, (0, 0, 0))
                    shadow_rect = shadow.get_rect(center=(cell_rect.centerx + 1, cell_rect.centery + 1))
                    
                    surface.blit(shadow, shadow_rect)
                    surface.blit(txt, txt_rect)

        for r in range(self.rows):
            for c in range(self.cols):
                val = board_array[r][c]
                if val == 0:
                    continue

                cell_rect = pygame.Rect(
                    self.rect.left + c * self.cell_w, self.rect.top + r * self.cell_h, self.cell_w, self.cell_h
                )

                color = CROSS_COLOR if val == 1 else CIRCLE_COLOR
                is_x = (val == 1 and not inverted) or (val == 2 and inverted)
                color = CROSS_COLOR if is_x else CIRCLE_COLOR
                if is_x:
                    ShapeDrawer.draw_x(surface, cell_rect, color)
                else:
                    ShapeDrawer.draw_o(surface, cell_rect, color)

    def get_cell_from_mouse(self, mouse_pos):
        if not self.rect.collidepoint(mouse_pos):
            return None

        local_x = mouse_pos[0] - self.rect.left
        local_y = mouse_pos[1] - self.rect.top

        row = local_y // self.cell_h
        col = local_x // self.cell_w

        row = max(0, min(int(row), self.rows - 1))
        col = max(0, min(int(col), self.cols - 1))

        return row, col

    def draw_ghost(self, surface, row, col, turn, inverted):
        """Dibuja el símbolo traslúcido en la celda donde está el mouse."""
        cell_rect = pygame.Rect(
            self.rect.left + col * self.cell_w, self.rect.top + row * self.cell_h, self.cell_w, self.cell_h
        )

        ghost_surf = pygame.Surface((self.cell_w, self.cell_h), pygame.SRCALPHA)
        local_rect = pygame.Rect(0, 0, self.cell_w, self.cell_h)

        is_x = (turn == 1 and not inverted) or (turn == 2 and inverted)
        color = (*CROSS_COLOR, 100) if is_x else (*CIRCLE_COLOR, 100)

        if is_x:
            ShapeDrawer.draw_x(ghost_surf, local_rect, color)
        else:
            ShapeDrawer.draw_o(ghost_surf, local_rect, color)

        surface.blit(ghost_surf, cell_rect.topleft)
