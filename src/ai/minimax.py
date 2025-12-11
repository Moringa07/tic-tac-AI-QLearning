# src/ai/minimax.py

import math
from copy import deepcopy
from typing import Dict, List, Tuple

from src.game_logic.board import Board

AI_PLAYER = 2
HUMAN_PLAYER = 1


def minimax_bruteforce(
    board: Board, depth: int, is_maximizing: bool, counter: Dict[str, int]
) -> int:
    """Versión original, cuenta los nodos evaluados."""
    counter["nodes"] += 1  # Incrementa el contador en cada llamada
    if board.winner == AI_PLAYER:
        return 1
    if board.winner == HUMAN_PLAYER:
        return -1
    if board.is_full():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in board.get_available_moves():
            temp_board = deepcopy(board)
            temp_board.make_move(move[0], move[1])
            score = minimax_bruteforce(
                temp_board, depth + 1, False, counter
            )  # Pasa el contador
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for move in board.get_available_moves():
            temp_board = deepcopy(board)
            temp_board.make_move(move[0], move[1])
            score = minimax_bruteforce(
                temp_board, depth + 1, True, counter
            )  # Pasa el contador
            best_score = min(score, best_score)
        return best_score


def minimax_alpha_beta(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    counter: Dict[str, int],
) -> int:
    """Versión optimizada del min-max"""
    counter["nodes"] += 1  # Incrementa el contador en cada llamada
    if board.winner == AI_PLAYER:
        return 1
    if board.winner == HUMAN_PLAYER:
        return -1
    if board.is_full():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in board.get_available_moves():
            temp_board = deepcopy(board)
            temp_board.make_move(move[0], move[1])
            score = minimax_alpha_beta(
                temp_board, depth + 1, alpha, beta, False, counter
            )  # Pasa el contador
            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score
    else:
        best_score = math.inf
        for move in board.get_available_moves():
            temp_board = deepcopy(board)
            temp_board.make_move(move[0], move[1])
            score = minimax_alpha_beta(
                temp_board, depth + 1, alpha, beta, True, counter
            )  # Pasa el contador
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score


def find_best_move_bruteforce(board: Board) -> Tuple[Tuple[int, int], List[dict]]:
    best_score = -math.inf
    best_move = board.get_available_moves()[0]
    graph_data = []
    for move in board.get_available_moves():
        temp_board = deepcopy(board)
        temp_board.make_move(move[0], move[1])
        score = minimax_bruteforce(
            temp_board, 0, False, {"nodes": 0}
        )  # Llama con un contador dummy
        graph_data.append(
            {"move": move, "score": score, "board": temp_board.board.tolist()}
        )
        if score > best_score:
            best_score, best_move = score, move
    return best_move, graph_data


def find_best_move_alpha_beta(board: Board) -> Tuple[Tuple[int, int], List[dict]]:
    best_score = -math.inf
    best_move = board.get_available_moves()[0]
    graph_data = []
    for move in board.get_available_moves():
        temp_board = deepcopy(board)
        temp_board.make_move(move[0], move[1])
        score = minimax_alpha_beta(
            temp_board, 0, -math.inf, math.inf, False, {"nodes": 0}
        )  # Llama con un contador dummy
        graph_data.append(
            {"move": move, "score": score, "board": temp_board.board.tolist()}
        )
        if score > best_score:
            best_score, best_move = score, move
    return best_move, graph_data


def get_simulation_move_bruteforce(board: Board) -> Tuple[Tuple[int, int], int]:
    """Retorna (mejor_movimiento, total_nodos_evaluados) para la simulación."""
    best_score = -math.inf
    best_move = board.get_available_moves()[0]
    total_nodes = 0
    for move in board.get_available_moves():
        temp_board = deepcopy(board)
        temp_board.make_move(move[0], move[1])
        counter = {"nodes": 0}
        score = minimax_bruteforce(temp_board, 0, False, counter)
        total_nodes += counter["nodes"]
        if score > best_score:
            best_score = score
            best_move = move
    return best_move, total_nodes


def get_simulation_move_alpha_beta(board: Board) -> Tuple[Tuple[int, int], int]:
    """
    Realiza una búsqueda unificada con poda Alfa-Beta.
    Retorna (mejor_movimiento, total_nodos_evaluados).
    """
    best_score = -math.inf
    best_move = board.get_available_moves()[0]
    counter = {"nodes": 0}  # Contamos el nodo raíz
    alpha, beta = -math.inf, math.inf

    for move in board.get_available_moves():
        temp_board = deepcopy(board)
        temp_board.make_move(move[0], move[1])

        score = minimax_alpha_beta(temp_board, 0, alpha, beta, False, counter)

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)

    return best_move, counter["nodes"]
