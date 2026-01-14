import math
from copy import deepcopy
from typing import Dict, List, Tuple

from src.game_logic.board import Board


def minimax_bruteforce(
    board: Board,
    depth: int,
    is_maximizing: bool,
    counter: Dict[str, int],
    maximizing_player_id: int,
) -> int:
    counter["nodes"] += 1

    if board.winner is not None and board.winner != 0:
        return 1 if board.winner == maximizing_player_id else -1

    if board.is_full():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in board.get_available_moves():
            prev_state = (board.turn, board.winner, board.game_over, board.win_info)

            board.make_move(move[0], move[1])

            score = minimax_bruteforce(board, depth + 1, False, counter, maximizing_player_id)

            board.undo_move(move[0], move[1], *prev_state)

            best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for move in board.get_available_moves():
            prev_state = (board.turn, board.winner, board.game_over, board.win_info)

            board.make_move(move[0], move[1])

            score = minimax_bruteforce(board, depth + 1, True, counter, maximizing_player_id)

            board.undo_move(move[0], move[1], *prev_state)

            best_score = min(score, best_score)
        return best_score


def minimax_alpha_beta(
    board: Board,
    depth: int,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    counter: Dict[str, int],
    maximizing_player_id: int,
) -> int:
    counter["nodes"] += 1

    if board.winner is not None and board.winner != 0:
        return 1 if board.winner == maximizing_player_id else -1
    if board.is_full():
        return 0

    if is_maximizing:
        best_score = -math.inf
        for move in board.get_available_moves():
            prev_state = (board.turn, board.winner, board.game_over, board.win_info)

            board.make_move(move[0], move[1])

            score = minimax_alpha_beta(board, depth + 1, alpha, beta, False, counter, maximizing_player_id)

            board.undo_move(move[0], move[1], *prev_state)

            best_score = max(score, best_score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score
    else:
        best_score = math.inf
        for move in board.get_available_moves():
            prev_state = (board.turn, board.winner, board.game_over, board.win_info)
            board.make_move(move[0], move[1])
            score = minimax_alpha_beta(board, depth + 1, alpha, beta, True, counter, maximizing_player_id)
            board.undo_move(move[0], move[1], *prev_state)
            best_score = min(score, best_score)
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_score


def find_best_move_bruteforce(
    board: Board,
) -> Tuple[Tuple[int, int], List[dict]]:
    best_score = -math.inf
    best_move = None
    graph_data = []

    ai_player_id = board.turn
    available_moves = board.get_available_moves()

    if not available_moves:
        return (0, 0), []

    best_move = available_moves[0]

    for move in available_moves:
        prev_state = (board.turn, board.winner, board.game_over, board.win_info)

        board.make_move(move[0], move[1])

        current_board_snapshot = [row[:] for row in board.board]

        score = minimax_bruteforce(board, 0, False, {"nodes": 0}, ai_player_id)

        board.undo_move(move[0], move[1], *prev_state)

        graph_data.append({"move": move, "score": score, "board": current_board_snapshot})

        if score > best_score:
            best_score, best_move = score, move

    return best_move, graph_data


def find_best_move_alpha_beta(board: Board) -> Tuple[Tuple[int, int], List[dict]]:
    best_score = -math.inf
    best_move = None
    graph_data = []
    ai_player_id = board.turn

    available_moves = board.get_available_moves()
    if not available_moves:
        return (0, 0), []

    for move in available_moves:
        prev_state = (board.turn, board.winner, board.game_over, board.win_info)

        board.make_move(move[0], move[1])

        current_board_snapshot = [row[:] for row in board.board]

        score = minimax_alpha_beta(board, 0, -math.inf, math.inf, False, {"nodes": 0}, ai_player_id)

        board.undo_move(move[0], move[1], *prev_state)

        graph_data.append({"move": move, "score": score, "board": current_board_snapshot})
        if score > best_score:
            best_score, best_move = score, move

    return best_move, graph_data


def get_focused_tree(
    board: Board,
    ai_player_id: int,
    scoring_function,
    current_depth=0,
    max_viz_depth=3,
):
    # Caso base: Juego terminado o profundidad máxima alcanzada
    if board.game_over or current_depth >= max_viz_depth:
        score = 0
        if board.winner == ai_player_id:
            score = 1
        elif board.winner is not None and board.winner != 0:
            score = -1
        return {
            "score": score,
            "board_matrix": [row[:] for row in board.board],  # Correcto: es una lista
            "children": [],
            "move": None,
        }

    is_maximizing = board.turn == ai_player_id
    candidates = []
    moves = board.get_available_moves()

    if not moves:
        return {
            "score": 0,
            "board_matrix": [row[:] for row in board.board],
            "children": [],
            "move": None,
        }

    for move in moves:
        # Aquí seguimos usando deepcopy porque esta función
        # se ejecuta pocas veces (solo para visualización)
        temp_board = deepcopy(board)
        temp_board.make_move(move[0], move[1])

        if scoring_function.__name__ == "minimax_bruteforce":
            score = scoring_function(temp_board, 0, not is_maximizing, {"nodes": 0}, ai_player_id)
        else:
            score = scoring_function(
                temp_board,
                0,
                -math.inf,
                math.inf,
                not is_maximizing,
                {"nodes": 0},
                ai_player_id,
            )

        candidates.append({"move": move, "score": score, "board": temp_board})

    # Elegir mejor candidato para el camino principal
    if is_maximizing:
        best_candidate = max(candidates, key=lambda x: x["score"])
    else:
        best_candidate = min(candidates, key=lambda x: x["score"])

    # Nodo actual
    node = {
        "score": best_candidate["score"],
        "board_matrix": [row[:] for row in board.board],
        "children": [],
        "best_move_coordinate": best_candidate["move"],
    }

    for cand in candidates:
        is_best_path = cand["move"] == best_candidate["move"]

        if is_best_path:
            child_node = get_focused_tree(
                cand["board"],  # Aquí pasamos el objeto Board porque la recursión lo necesita
                ai_player_id,
                scoring_function,
                current_depth + 1,
                max_viz_depth,
            )
            child_node["score"] = cand["score"]
            child_node["is_chosen"] = True
            child_node["move"] = cand["move"]
            node["children"].append(child_node)
        else:
            # --- EL ERROR ESTABA AQUÍ ---
            leaf_node = {
                "score": cand["score"],
                # Extraemos la matriz (.board) y la copiamos ([row[:]])
                "board_matrix": [row[:] for row in cand["board"].board],
                "children": [],
                "is_chosen": False,
                "move": cand["move"],
            }
            node["children"].append(leaf_node)

    return node


def find_best_move_and_viz(board: Board, use_alpha_beta: bool):
    """
    Calcula el mejor movimiento y genera el árbol visual.
    Retorna: (best_move, tree_root_node)
    """
    ai_player_id = board.turn

    scoring_func = minimax_alpha_beta if use_alpha_beta else minimax_bruteforce

    root_node = get_focused_tree(board, ai_player_id, scoring_func, max_viz_depth=3)

    best_move = root_node.get("best_move_coordinate")

    return best_move, root_node


def get_simulation_move_bruteforce(board: Board) -> Tuple[Tuple[int, int], int]:
    """Retorna (mejor_movimiento, total_nodos_evaluados) para la simulación."""

    ai_player_id = board.turn

    moves = board.get_available_moves()
    if not moves:
        return (0, 0), 0

    best_score = -math.inf
    best_move = moves[0]
    total_nodes = 0

    for move in moves:
        temp_board = deepcopy(board)
        temp_board.make_move(move[0], move[1])
        counter = {"nodes": 0}

        score = minimax_bruteforce(temp_board, 0, False, counter, ai_player_id)

        total_nodes += counter["nodes"]
        if score > best_score:
            best_score = score
            best_move = move

    return best_move, total_nodes


def get_simulation_move_alpha_beta(board: Board) -> Tuple[Tuple[int, int], int]:
    """
    Retorna (mejor_movimiento, total_nodos_evaluados). Para la simulación
    """

    ai_player_id = board.turn

    moves = board.get_available_moves()
    if not moves:
        return (0, 0), 0

    best_score = -math.inf
    best_move = moves[0]
    counter = {"nodes": 0}
    alpha, beta = -math.inf, math.inf

    for move in moves:
        temp_board = deepcopy(board)
        temp_board.make_move(move[0], move[1])

        score = minimax_alpha_beta(temp_board, 0, alpha, beta, False, counter, ai_player_id)

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)

    return best_move, counter["nodes"]
