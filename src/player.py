import numpy as np
from Board import valid_moves, make_move, check_win

def random_player(board, _player):
    """Choose a valid move randomly."""
    return np.random.choice(valid_moves(board))

def heuristic_player(board, player):
    """Try to win, block opponent, or play center."""
    opponent = 3 - player
    moves = valid_moves(board)
    # Win if possible
    for move in moves:
        temp = board.copy()
        make_move(temp, move, player)
        if check_win(temp, player):
            return move
    # Block opponent
    for move in moves:
        temp = board.copy()
        make_move(temp, move, opponent)
        if check_win(temp, opponent):
            return move
    # Take center if free
    center = 3
    if center in moves:
        return center
    # Random move otherwise
    return np.random.choice(moves)

def intelligent_player(board, player, depth=1):
    """Minimax-based player with given depth."""
    opponent = 3 - player
    best_score = -np.inf
    best_move = valid_moves(board)[0]
    for move in valid_moves(board):
        temp = board.copy()
        make_move(temp, move, player)
        score = minimax(temp, depth-1, False, player, opponent)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move

def minimax(board, depth, maximizing, player, opponent):
    """Recursive minimax evaluation."""
    if check_win(board, player):
        return 100
    if check_win(board, opponent):
        return -100
    if depth == 0 or len(valid_moves(board)) == 0:
        return evaluate_board(board, player)
    if maximizing:
        return max(minimax(apply_move(board, move, player), depth-1, False, player, opponent)
                   for move in valid_moves(board))
    else:
        return min(minimax(apply_move(board, move, opponent), depth-1, True, player, opponent)
                   for move in valid_moves(board))

def apply_move(board, col, player):
    new_board = board.copy()
    make_move(new_board, col, player)
    return new_board

def evaluate_board(board, player):
    """Score board for minimax."""
    opponent = 3 - player
    score = 0
    for r in range(6):
        for c in range(4):
            window = board[r, c:c+4]
            score += window_score(window, player, opponent)
    return score

def window_score(window, player, opponent):
    """Score a window of 4 cells."""
    score = 0
    if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 1:
        score += 10
    if np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == 0) == 1:
        score -= 8
    return score
