import pygame


class ShapeDrawer:
    @staticmethod
    def draw_x(surface, rect, color, width_weight=0.1):
        """
        Dibuja una X dentro de cualquier pygame.Rect.
        width_weight: grosor de la línea proporcional al tamaño del rect.
        """
        margin = rect.width // 4

        line_width = max(1, int(rect.width * width_weight))

        start_desc = (rect.left + margin, rect.top + margin)
        end_desc = (rect.right - margin, rect.bottom - margin)

        start_asc = (rect.left + margin, rect.bottom - margin)
        end_asc = (rect.right - margin, rect.top + margin)

        pygame.draw.line(surface, color, start_desc, end_desc, line_width)
        pygame.draw.line(surface, color, start_asc, end_asc, line_width)

    @staticmethod
    def draw_o(surface, rect, color, width_weight=0.1):
        """
        Dibuja un círculo dentro de cualquier pygame.Rect.
        """
        padding = rect.width // 6
        radius = (rect.width // 2) - padding
        center = rect.center

        line_width = max(1, int(rect.width * width_weight))

        pygame.draw.circle(surface, color, center, radius, line_width)
