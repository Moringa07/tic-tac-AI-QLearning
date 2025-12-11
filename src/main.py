import sys
import time
import random

import pygame

from src.ai.minimax import find_best_move_alpha_beta, find_best_move_bruteforce
from src.config import *
from src.game_logic.board import Board
from src.gui.renderer import Renderer

HUMAN = "HUMAN"
AI_SLOW = "AI_SLOW"
AI_FAST = "AI_FAST"
AI_START_SELECTION = "AI_START_SELECTION"


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
    game_state = "MENU"
    player_types = None
    last_graph_data = []  

    menu_options = [
        "Humano vs Humano",
        "Humano vs IA (Lenta - Minimax)",
        "Humano vs IA (Rápida - AlfaBeta)",
        "IA Lenta vs IA Rápida (¡Observa!)",
    ]
    selected_option = 0
    
    # Variables del submenú de selección inicial
    ai_speed_selected = None
    start_selection_options = ["Humano Inicia", "IA Inicia"]
    start_selected_option = 0
    
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
                        if selected_option == 0:
                            player_types = [HUMAN, HUMAN]
                            game_state = "PLAYING"
                        elif selected_option == 1:
                            ai_speed_selected = AI_SLOW
                            game_state = AI_START_SELECTION
                        elif selected_option == 2:
                            ai_speed_selected = AI_FAST
                            game_state = AI_START_SELECTION
                        elif selected_option == 3:
                            player_types = [AI_SLOW, AI_FAST]
                            game_state = "PLAYING"
                        
                        waiting_for_step = False
                        pygame.event.clear()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: 
                        for i, rect in enumerate(menu_option_rects):
                            if rect.collidepoint(mouse_pos):
                                selected_option = i
                                if selected_option == 0:
                                    player_types = [HUMAN, HUMAN]
                                    game_state = "PLAYING"
                                elif selected_option == 1:
                                    ai_speed_selected = AI_SLOW
                                    game_state = AI_START_SELECTION
                                elif selected_option == 2:
                                    ai_speed_selected = AI_FAST
                                    game_state = AI_START_SELECTION
                                elif selected_option == 3:
                                    player_types = [AI_SLOW, AI_FAST]
                                    game_state = "PLAYING"
                                
                                board = Board()
                                last_graph_data = []
                                pygame.event.clear()
                                waiting_for_step = False

            for i, rect in enumerate(menu_option_rects):
                if rect.collidepoint(mouse_pos):
                    selected_option = i

            menu_option_rects = renderer.draw_menu(menu_options, selected_option)

        elif game_state == AI_START_SELECTION:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        start_selected_option = (start_selected_option - 1) % len(start_selection_options)
                    elif event.key == pygame.K_DOWN:
                        start_selected_option = (start_selected_option + 1) % len(start_selection_options)
                    elif event.key == pygame.K_RETURN:
                        p1 = HUMAN
                        p2 = ai_speed_selected
                        
                        selection = start_selection_options[start_selected_option]
                        
                        if selection == "Humano Inicia":
                            player_types = [p1, p2]
                        elif selection == "IA Inicia":
                            player_types = [p2, p1]
                        
                        board = Board()
                        last_graph_data = []
                        waiting_for_step = False
                        game_state = "PLAYING"
                        pygame.event.clear()

                    elif event.key == pygame.K_ESCAPE:
                        game_state = "MENU"

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        for i, rect in enumerate(menu_option_rects):
                            if rect.collidepoint(mouse_pos):
                                start_selected_option = i
                                p1 = HUMAN
                                p2 = ai_speed_selected
                                selection = start_selection_options[start_selected_option]
                                
                                if selection == "Humano Inicia":
                                    player_types = [p1, p2]
                                elif selection == "IA Inicia":
                                    player_types = [p2, p1]
                                
                                board = Board()
                                last_graph_data = []
                                waiting_for_step = False
                                game_state = "PLAYING"
                                pygame.event.clear()
            
            for i, rect in enumerate(menu_option_rects):
                if rect.collidepoint(mouse_pos):
                    start_selected_option = i
            
            menu_option_rects = renderer.draw_menu(start_selection_options, start_selected_option)

        # --- Lógica de la Partida ---
        elif game_state == "PLAYING":
            current_player_type = player_types[board.turn - 1]

            reset_triggered = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        board = Board()  
                        last_graph_data = []
                        pygame.event.clear()
                        waiting_for_step = False
                        reset_triggered = True
                    if event.key == pygame.K_ESCAPE:
                        game_state = "MENU"

                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and not board.game_over
                    and current_player_type == HUMAN
                ):
                    mouseX, mouseY = event.pos
                    # Ajustar coordenadas con OFFSET
                    if (
                        renderer.board_offset_x
                        <= mouseX
                        < renderer.board_offset_x + BOARD_WIDTH
                        and mouseY > BOARD_OFFSET_Y
                    ):
                        clicked_row = (mouseY - BOARD_OFFSET_Y) // SQUARE_SIZE
                        clicked_col = (mouseX - renderer.board_offset_x) // SQUARE_SIZE

                        if (
                            0 <= clicked_row < BOARD_ROWS
                            and 0 <= clicked_col < BOARD_COLS
                        ):
                            board.make_move(clicked_row, clicked_col)

            # Lógica para llamar a la IA correcta
            # Verificación de centrado
            is_h_vs_h = player_types == [HUMAN, HUMAN]
            renderer.set_centered(is_h_vs_h)
            
            # Verificación de inversión de símbolos
            # Si el Humano es el Jugador 2, se espera que sea X (Visualmente) 
            # Excepción en IA vs IA nada más
            should_invert = False
            if len(player_types) == 2:
                if player_types[1] == HUMAN and player_types[0] != HUMAN:
                     should_invert = True
            
            renderer.set_inverted(should_invert)

            # Step-by-step Wait Logic for AI vs AI
            is_ai_vs_ai = player_types == [AI_SLOW, AI_FAST]

            if is_ai_vs_ai and waiting_for_step:
                font_prompt = pygame.font.Font(None, 40)
                prompt_surf = font_prompt.render(
                    "Presiona ENTER para siguiente jugada...", True, (255, 255, 0)
                )
                prompt_rect = prompt_surf.get_rect(center=(WIDTH // 2, HEIGHT - 50))

                bg_rect = prompt_rect.inflate(20, 10)
                pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                screen.blit(prompt_surf, prompt_rect)

                pygame.display.update()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RETURN:
                            waiting_for_step = False
                        elif event.key == pygame.K_r: 
                            board = Board()
                            last_graph_data = []
                            pygame.event.clear()
                            waiting_for_step = False
                            reset_triggered = True
                        if event.key == pygame.K_ESCAPE:
                            game_state = "MENU"
                continue  

            if (
                not reset_triggered
                and not board.game_over
                and current_player_type in [AI_FAST, AI_SLOW]
            ):
                start_time = time.time()
                move = None
                pygame.display.update()  

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
                        waiting_for_step = True  # Pausa después de este movimiento

            # Dibujado
            renderer.draw_grid()
            renderer.draw_symbols(board.board)
            renderer.draw_decision_graph(last_graph_data)

            # Indicador de Turno
            # Recalcular el tipo de jugador actual para el dibujado (por si cambió post-jugada)
            real_current_player = player_types[board.turn - 1]
            player_label = "HUMAN" if real_current_player == HUMAN else "IA"
            renderer.draw_turn_indicator(board.turn, player_label)

            # Símbolo Fantasma (Hover)
            if not board.game_over and current_player_type == HUMAN:
                mouseX, mouseY = mouse_pos
                if (
                    renderer.board_offset_x
                    <= mouseX
                    < renderer.board_offset_x + BOARD_WIDTH
                    and mouseY > BOARD_OFFSET_Y
                ):
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
