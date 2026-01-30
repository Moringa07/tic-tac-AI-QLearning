import pygame
from src.config import FONT_COLOR, MENU_SELECTED_COLOR, BG_COLOR, LINE_COLOR, SQUARE_HOVER_COLOR, CIRCLE_COLOR, MENU_BUTTON_COLOR

class ConfigComponent:
    def __init__(self, rect, font):
        self.rect = rect
        self.font = font
        
        # Parámetros configurables
        self.sliders = [
            {"label": "Alpha (Aprendizaje)", "min": 0.1, "max": 1.0, "val": 0.5, "step": 0.01},
            {"label": "Gamma (Descuento)",   "min": 0.1, "max": 1.0, "val": 0.9, "step": 0.01},
            {"label": "Epsilon (Exploración)", "min": 0.0, "max": 1.0, "val": 0.1, "step": 0.01},
            {"label": "Episodios",           "min": 1000, "max": 50000, "val": 10000, "step": 1000},
        ]
        
        self.dragging_slider = None
        self.start_button_rect = pygame.Rect(0, 0, 200, 50)
        self.start_button_rect.centerx = rect.centerx
        self.start_button_rect.bottom = rect.bottom - 20

    def draw(self, surface):
        surface.fill(BG_COLOR)

        # Contenedor central simulado
        total_height = 80 + (len(self.sliders) * 80) + 80 # titulo + sliders + boton
        # Centrado vertical puro con un pequeño ajuste visual hacia arriba (-40)
        start_y_block = (self.rect.height - total_height) // 2 - 40

        # Título
        title = self.font.render("Configura tu Agente", True, FONT_COLOR)
        title_rect = title.get_rect(center=(self.rect.centerx, start_y_block))
        surface.blit(title, title_rect)
        
        # Línea decorativa
        pygame.draw.line(surface, LINE_COLOR, 
                         (title_rect.left - 20, title_rect.bottom + 10),
                         (title_rect.right + 20, title_rect.bottom + 10), 4)
        
        
        # Sliders
        # Aumentamos la distancia entre título y primera opción
        start_y_sliders = title_rect.bottom + 80
        slider_width = 400
        slider_height = 10
        
        for i, slider in enumerate(self.sliders):
            y = start_y_sliders + i * 80
            
            # Etiqueta
            label_name = slider['label']
            val_display = f"{slider['val']:.2f}"
            if "Episodios" in label_name:
                 val_display = f"{int(slider['val'])}"
            
            label_surf = self.font.render(f"{label_name}: {val_display}", True, FONT_COLOR)
            label_rect = label_surf.get_rect(center=(self.rect.centerx, y - 25))
            surface.blit(label_surf, label_rect)
            
            # Barra
            bar_rect = pygame.Rect(0, 0, slider_width, slider_height)
            bar_rect.center = (self.rect.centerx, y + 10)
            pygame.draw.rect(surface, SQUARE_HOVER_COLOR, bar_rect, border_radius=5)
            
            # Relleno
            ratio = (slider["val"] - slider["min"]) / (slider["max"] - slider["min"])
            fill_width = ratio * slider_width
            fill_rect = pygame.Rect(bar_rect.left, bar_rect.top, fill_width, slider_height)
            pygame.draw.rect(surface, LINE_COLOR, fill_rect, border_radius=5)
            
            # Handle
            handle_x = bar_rect.left + fill_width
            handle_center = (int(handle_x), int(y + 10))
            
            color = MENU_SELECTED_COLOR if self.dragging_slider == i else CIRCLE_COLOR
            pygame.draw.circle(surface, color, handle_center, 12)
            pygame.draw.circle(surface, (255, 255, 255), handle_center, 4)
            
            slider["bar_rect"] = bar_rect

        # Botón de Iniciar
        self.start_button_rect.centerx = self.rect.centerx
        self.start_button_rect.top = start_y_sliders + len(self.sliders) * 80 + 20
        
        pygame.draw.rect(surface, MENU_BUTTON_COLOR, self.start_button_rect, border_radius=15)
        pygame.draw.rect(surface, LINE_COLOR, self.start_button_rect, 2, border_radius=15)
        
        # Texto del botón
        btn_txt = self.font.render("Entrenar", True, FONT_COLOR)
        surface.blit(btn_txt, btn_txt.get_rect(center=self.start_button_rect.center))

        # Texto ESC
        esc_txt = pygame.font.Font(None, 24).render("ESC para volver", True, (100, 120, 120))
        surface.blit(esc_txt, esc_txt.get_rect(center=(self.rect.centerx, self.rect.bottom - 30)))
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            
            if self.start_button_rect.collidepoint(mouse_pos):
                return self.get_values()
            
            for i, slider in enumerate(self.sliders):
                detect_rect = slider["bar_rect"].inflate(20, 30)
                if detect_rect.collidepoint(mouse_pos):
                    self.dragging_slider = i
                    self._update_slider_value(i, mouse_pos[0])
                    break
                    
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging_slider = None
            
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_slider is not None:
                self._update_slider_value(self.dragging_slider, event.pos[0])
                
        return None

    def _update_slider_value(self, index, mouse_x):
        slider = self.sliders[index]
        rect = slider["bar_rect"]
        
        ratio = (mouse_x - rect.left) / rect.width
        ratio = max(0.0, min(1.0, ratio))
        
        new_val = slider["min"] + ratio * (slider["max"] - slider["min"])
        
        if "Episodios" in slider["label"]:
             new_val = round(new_val / 1000) * 1000
        
        slider["val"] = new_val

    def get_values(self):
        return {
            "alpha": self.sliders[0]["val"],
            "gamma": self.sliders[1]["val"],
            "epsilon": self.sliders[2]["val"],
            "episodes": int(self.sliders[3]["val"])
        }
