import sys
import time

import pygame

from src.ai.minimax import find_best_move_alpha_beta, find_best_move_bruteforce
from src.config import *
from src.game_logic.board import Board
from src.gui.renderer import Renderer

HUMAN = "HUMAN"
AI_SLOW = "AI_SLOW"
AI_FAST = "AI_FAST"


def main_game_loop():
    # --- Inicialización ---
    pygame.display.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Comparador de IA: Minimax vs Alfa-Beta")
    clock = pygame.time.Clock()

    board = Board()
    renderer = Renderer(screen)

    # --- Variables de Estado y Menú Ampliado ---
    # --- Variables de Estado y Menú Ampliado ---
    game_state = "MENU"
    player_types = None
    last_graph_data = [] # Store graph data to display

    menu_options = [
        "Humano vs Humano",
        "Humano vs IA (Lenta - Minimax)",
        "Humano vs IA (Rápida - AlfaBeta)",
        "IA Lenta vs IA Rápida (¡Observa!)",
    ]
    selected_option = 0
    running = True
    menu_option_rects = []

    while running:
        mouse_pos = pygame.mouse.get_pos()
        
        if game_state == "MENU":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Input de Teclado
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_option = (selected_option - 1) % len(menu_options)
                    elif event.key == pygame.K_DOWN:
                        selected_option = (selected_option + 1) % len(menu_options)
                    elif event.key == pygame.K_RETURN:
                        # Logic duplicated below, extracted for cleanliness ideally but inline is fine
                        if selected_option == 0: player_types = [HUMAN, HUMAN]
                        elif selected_option == 1: player_types = [HUMAN, AI_SLOW]
                        elif selected_option == 2: player_types = [HUMAN, AI_FAST]
                        game_state = "PLAYING"
                        waiting_for_step = False # Initialize step flag
                        pygame.event.clear()

                # Input de Mouse (Clic)
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Clic izquierdo
                        for i, rect in enumerate(menu_option_rects):
                            if rect.collidepoint(mouse_pos):
                                selected_option = i
                                # Trigger selection
                                if selected_option == 0: player_types = [HUMAN, HUMAN]
                                elif selected_option == 1: player_types = [HUMAN, AI_SLOW]
                                elif selected_option == 2: player_types = [HUMAN, AI_FAST]
                                elif selected_option == 3: player_types = [AI_SLOW, AI_FAST]
                                board = Board()
                                last_graph_data = [] 
                                game_state = "PLAYING"
                                pygame.event.clear()
                                waiting_for_step = False

            # Lógica Mouse Hover
            for i, rect in enumerate(menu_option_rects):
                if rect.collidepoint(mouse_pos):
                    selected_option = i

            menu_option_rects = renderer.draw_menu(menu_options, selected_option)

        # --- Lógica de la Partida ---
        elif game_state == "PLAYING":
            current_player_type = player_types[board.turn - 1]

            reset_triggered = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        board = Board() # Hard reset
                        last_graph_data = [] 
                        pygame.event.clear()
                        reset_triggered = True 
                        waiting_for_step = False
                        break 
                    if event.key == pygame.K_ESCAPE:
                        game_state = "MENU"

                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and not board.game_over
                    and current_player_type == HUMAN
                ):
                    mouseX, mouseY = event.pos
                    # Ajustar coordenadas con OFFSET
                    if (renderer.board_offset_x <= mouseX < renderer.board_offset_x + BOARD_WIDTH 
                        and mouseY > BOARD_OFFSET_Y):
                        clicked_row = (mouseY - BOARD_OFFSET_Y) // SQUARE_SIZE
                        clicked_col = (mouseX - renderer.board_offset_x) // SQUARE_SIZE
                        
                        if 0 <= clicked_row < BOARD_ROWS and 0 <= clicked_col < BOARD_COLS:
                            board.make_move(clicked_row, clicked_col)

            # Lógica para llamar a la IA correcta
            # Centering Check
            is_h_vs_h = (player_types == [HUMAN, HUMAN])
            renderer.set_centered(is_h_vs_h)
            
            # Step-by-step Wait Logic for AI vs AI
            is_ai_vs_ai = (player_types == [AI_SLOW, AI_FAST])
            
            if is_ai_vs_ai and waiting_for_step:
                # Draw "Press Enter" prompt
                font_prompt = pygame.font.Font(None, 40)
                prompt_surf = font_prompt.render("Presiona ENTER para siguiente jugada...", True, (255, 255, 0))
                prompt_rect = prompt_surf.get_rect(center=(WIDTH // 2, HEIGHT - 50))
                
                # Small background for readability
                bg_rect = prompt_rect.inflate(20, 10)
                pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                screen.blit(prompt_surf, prompt_rect)
                
                pygame.display.update()
                
                # Event loop specifically for waiting
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            waiting_for_step = False
                        elif event.key == pygame.K_r: # Handle Restart during pause
                             board = Board()
                             last_graph_data = []
                             pygame.event.clear()
                             waiting_for_step = False
                             # We don't need 'break' here to exit main loop, just 'continue' leads to top
                             # But we need to skip the rendering of the old board potentially?
                             # Actually setting waiting_for_step=False will allow loop to continue.
                             # But we want to 'restart'.
                             # We should probably set reset_triggered or similar, or just let it loop.
                             # If we set waiting_for_step = False, it continues to 'continue'.
                             # Then loop wrapper runs.
                             # Next iteration: game_state=PLAYING.
                             # current_player = 0.
                             # It will draw new board.
                             # Then AI logic will run.
                             # If we want to skip AI logic first frame (like main R):
                             reset_triggered = True 
                        if event.key == pygame.K_ESCAPE:
                            game_state = "MENU"
                continue # Skip the rest of loop until step is taken

            if not reset_triggered and not board.game_over and current_player_type in [AI_SLOW, AI_FAST]:
                start_time = time.time()
                move = None
                pygame.display.update() # Forzar dibujado antes de pensar

                if current_player_type == AI_SLOW:
                    print(f"Turno {board.turn} (IA Lenta): Calculando movimiento...")
                    move, graph_data = find_best_move_bruteforce(board)
                    last_graph_data = graph_data

                elif current_player_type == AI_FAST:
                    print(f"Turno {board.turn} (IA Rápida): Calculando movimiento...")
                    move, graph_data = find_best_move_alpha_beta(board)
                    last_graph_data = graph_data

                end_time = time.time()
                print(f"Cálculo completado en {end_time - start_time:.4f} segundos.")

                if move:
                    board.make_move(move[0], move[1])
                    if is_ai_vs_ai:
                        waiting_for_step = True # Pause after this move

            # Dibujado
            renderer.draw_grid()
            renderer.draw_symbols(board.board)
            renderer.draw_decision_graph(last_graph_data)
            
            # Indicador de Turno
            # Recalcular el tipo de jugador actual para el dibujado (por si cambió post-jugada)
            real_current_player = player_types[board.turn - 1]
            player_label = "HUMAN" if real_current_player == HUMAN else "IA"
            renderer.draw_turn_indicator(board.turn, player_label)

            # Ghost Symbol (Hover)
            if not board.game_over and current_player_type == HUMAN:
                mouseX, mouseY = mouse_pos
                if (renderer.board_offset_x <= mouseX < renderer.board_offset_x + BOARD_WIDTH 
                    and mouseY > BOARD_OFFSET_Y):
                    hover_row = (mouseY - BOARD_OFFSET_Y) // SQUARE_SIZE
                    hover_col = (mouseX - renderer.board_offset_x) // SQUARE_SIZE
                    if 0 <= hover_row < BOARD_ROWS and 0 <= hover_col < BOARD_COLS:
                         if board.is_valid_move(hover_row, hover_col):
                             renderer.draw_ghost_symbol(hover_row, hover_col, board.turn)

            if board.game_over:
                renderer.draw_win_line(board)
                renderer.draw_game_over_text(board)

        # --- Actualización de Pantalla ---
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main_game_loop()
