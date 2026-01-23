# simulate.py

import csv
import random
import time
from typing import Dict, List, Tuple

# Importamos las nuevas funciones específicas para la simulación
from ai.minimax import (
    get_simulation_move_alpha_beta,
    get_simulation_move_bruteforce,
)
from game_logic.board import Board

# --- CONFIGURACIÓN DEL EXPERIMENTO ---
NUM_SIMULATIONS_PER_BATCH = 10  # teorema del limite central tiende a dist normal
CSV_FILENAME = "queries/full_simulation_results.csv"

# Tipos de jugadores para la simulación
AI_SLOW = "Minimax"
AI_FAST = "Alpha-Beta"
RANDOM = "Random"


def get_ai_move(board: Board, player_type: str) -> Tuple[Tuple[int, int], int]:
    """Llama a la función de IA correcta y retorna el movimiento y los nodos."""
    if player_type == AI_SLOW:
        return get_simulation_move_bruteforce(board)
    elif player_type == AI_FAST:
        return get_simulation_move_alpha_beta(board)
    return ((-1, -1), 0)


def get_random_move(board: Board) -> Tuple[int, int]:
    """Elige un movimiento válido al azar."""
    return random.choice(board.get_available_moves())


def run_single_simulation(player1_type: str, player2_type: str) -> List[Dict]:
    """Simula una partida entre dos tipos de jugadores definidos."""
    board = Board()
    game_records = []
    turn_number = 1

    while not board.game_over:
        current_player = player1_type if (turn_number % 2 != 0) else player2_type

        start_time = time.time()
        nodes_evaluated = 0

        if current_player == RANDOM:
            move = get_random_move(board)
        else:  # Es una IA
            move, nodes_evaluated = get_ai_move(board, current_player)

        end_time = time.time()

        if move != (-1, -1):
            board.make_move(move[0], move[1])

        record = {
            "turn": turn_number,
            "pieces_on_board": turn_number - 1,
            "algorithm": current_player,
            "nodes_evaluated": nodes_evaluated,
            "time_seconds": end_time - start_time,
            "winner": board.winner if board.game_over else 0,
        }
        game_records.append(record)
        turn_number += 1

    return game_records


def main():
    all_results = []

    experiment_batches = {
        "Direct_Comparison": (AI_SLOW, AI_FAST),
        "Minimax_Profile": (AI_SLOW, RANDOM),
        "AlphaBeta_Profile": (AI_FAST, RANDOM),
    }

    total_sims = len(experiment_batches) * NUM_SIMULATIONS_PER_BATCH
    print(f"Iniciando {total_sims} simulaciones en {len(experiment_batches)} lotes...")
    sim_counter = 0
    for batch_name, (p1, p2) in experiment_batches.items():
        print(f"\n--- Ejecutando Lote: {batch_name} ({NUM_SIMULATIONS_PER_BATCH} partidas) ---")
        for i in range(NUM_SIMULATIONS_PER_BATCH):
            sim_counter += 1
            print(f"  - Simulando partida {sim_counter}/{total_sims}...", end="\r")

            results = run_single_simulation(p1, p2)
            for record in results:
                record["experiment_batch"] = batch_name
                record["simulation_id"] = f"{batch_name}_{i + 1}"
                all_results.append(record)
        print(f"\nLote {batch_name} completado.")

    print(f"\nSimulación finalizada. Guardando resultados en '{CSV_FILENAME}'...")
    if all_results:
        with open(CSV_FILENAME, "w", newline="") as csvfile:
            fieldnames = all_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
    print("¡Listo!")


if __name__ == "__main__":
    main()
