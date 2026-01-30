import pygame

from src.config import (
    FONT_COLOR, FONT_SIZE, MENU_SELECTED_COLOR, 
    BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_MARGIN, 
    MENU_BUTTON_COLOR, MENU_BUTTON_HOVER
)


class MenuComponent:
    def __init__(self, screen_rect, font, title_font):
        self.rect = screen_rect
        self.font = font
        self.title_font = title_font

    def draw(self, surface, options, selected_option):
        option_rects = []
        
        # Título
        title_surf = self.title_font.render("Tres en Raya", True, FONT_COLOR)
        # Posicionar título un poco más arriba para dejar espacio
        title_rect = title_surf.get_rect(center=(self.rect.centerx, self.rect.height * 0.15))
        surface.blit(title_surf, title_rect)

        # Configuración del layout del menú
        total_height = len(options) * (BUTTON_HEIGHT + BUTTON_MARGIN)
        start_y = (self.rect.height - total_height) // 2 + 50 # Centrado verticalmente, offset por título

        # Dibujar Opciones como Botones Simétricos
        for i, option in enumerate(options):
            y = start_y + i * (BUTTON_HEIGHT + BUTTON_MARGIN)
            cx = self.rect.centerx
            
            # Crear Rect del botón
            btn_rect = pygame.Rect(0, 0, BUTTON_WIDTH, BUTTON_HEIGHT)
            btn_rect.center = (cx, y)
            
            is_sel = i == selected_option
            
            # Colores
            bg_color = MENU_BUTTON_HOVER if is_sel else MENU_BUTTON_COLOR
            border_color = MENU_SELECTED_COLOR if is_sel else (100, 100, 120)
            text_color = MENU_SELECTED_COLOR if is_sel else FONT_COLOR

            # Dibujar botón (Fondo y Borde)
            pygame.draw.rect(surface, bg_color, btn_rect, border_radius=15)
            pygame.draw.rect(surface, border_color, btn_rect, width=3 if is_sel else 1, border_radius=15)

            # Efecto Glow si seleccionado
            if is_sel:
                glow_rect = btn_rect.inflate(6, 6)
                pygame.draw.rect(surface, (MENU_SELECTED_COLOR[0], MENU_SELECTED_COLOR[1], MENU_SELECTED_COLOR[2], 50), glow_rect, width=2, border_radius=18)

            # Texto
            txt_surf = self.font.render(option, True, text_color)
            txt_rect = txt_surf.get_rect(center=btn_rect.center)
            surface.blit(txt_surf, txt_rect)
            
            option_rects.append(btn_rect)

        return option_rects
