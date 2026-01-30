import pygame

from src.config import BG_COLOR, CIRCLE_COLOR, CROSS_COLOR, FONT_COLOR, LINE_WIDTH, WIN_LINE_COLOR


class HUDComponent:
    def __init__(self, font):
        self.font = font

    def draw_turn(self, surface, board_rect, turn, player_type, inverted):
        is_x = (turn == 1 and not inverted) or (turn == 2 and inverted)
        symbol = "X" if is_x else "O"
        color = CROSS_COLOR if is_x else CIRCLE_COLOR

        text_str = f"Turno de {'Humano' if player_type == 'HUMAN' else 'IA'} ({symbol})"
        txt_surf = self.font.render(text_str, True, color)

        # Posicionar arriba del tablero (más cerca)
        pos = (board_rect.centerx, board_rect.top - 50)
        surface.blit(txt_surf, txt_surf.get_rect(center=pos))

    def draw_win_line(self, surface, board_view, board_logic):
        """Dibuja la línea de victoria con efecto neón."""
        if not board_logic.win_info:
            return

        win_type, index = board_logic.win_info
        b_rect = board_view.rect

        # Puntos de inicio y fin (relativos al board_rect)
        if win_type == "row":
            y = b_rect.top + index * board_view.cell_h + board_view.cell_h // 2
            start_pos = (b_rect.left + 20, y)
            end_pos = (b_rect.right - 20, y)

        elif win_type == "col":
            x = b_rect.left + index * board_view.cell_w + board_view.cell_w // 2
            start_pos = (x, b_rect.top + 20)
            end_pos = (x, b_rect.bottom - 20)

        elif win_type == "diag":
            if index == 1:  # Descendente
                start_pos = (b_rect.left + 20, b_rect.top + 20)
                end_pos = (b_rect.right - 20, b_rect.bottom - 20)
            else:  # Ascendente
                start_pos = (b_rect.left + 20, b_rect.bottom - 20)
                end_pos = (b_rect.right - 20, b_rect.top + 20)

        glow_colors = [
            (*WIN_LINE_COLOR, 50),
            (*WIN_LINE_COLOR, 120),
            (*WIN_LINE_COLOR, 255),
        ]
        widths = [LINE_WIDTH + 15, LINE_WIDTH + 7, LINE_WIDTH]

        glow_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        for i in range(3):
            pygame.draw.line(glow_surf, glow_colors[i], start_pos, end_pos, widths[i])

        surface.blit(glow_surf, (0, 0))

    def draw_game_over(self, surface, screen_rect, board):
        if not board.game_over:
            return

        overlay = pygame.Surface((screen_rect.width, screen_rect.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        surface.blit(overlay, (0, 0))

        msg = "¡Empate!" if board.winner == 0 else f"¡Jugador {board.winner} Gana!"
        color = FONT_COLOR if board.winner == 0 else (WIN_LINE_COLOR)

        txt_surf = self.font.render(msg, True, color)
        txt_rect = txt_surf.get_rect(center=(screen_rect.centerx, screen_rect.centery - 40))
        surface.blit(txt_surf, txt_rect)

        # Instrucción de reinicio
        res_surf = self.font.render("Pulsa 'R' para reiniciar", True, (200, 200, 200))
        res_rect = res_surf.get_rect(center=(screen_rect.centerx, screen_rect.centery + 40))
        surface.blit(res_surf, res_rect)

    def draw_step_prompt(self, surface):
        """Dibuja el aviso de espera entre turnos de IA."""
        font_prompt = pygame.font.Font(None, 30)
        text_str = "Presione ENTER para siguiente jugada..."

        prompt = font_prompt.render(text_str, True, CIRCLE_COLOR)
        screen_rect = surface.get_rect()
        rect = prompt.get_rect(center=(screen_rect.width // 3.25, screen_rect.height - 25))

        bg_rect = rect.inflate(20, 10)
        pygame.draw.rect(surface, BG_COLOR, bg_rect)

        surface.blit(prompt, rect)
