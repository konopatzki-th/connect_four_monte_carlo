import numpy as np

NUM_ROWS = 6
NUM_COLS = 7

def create_board():
    """Create an empty 6x7 board."""
    return np.zeros((NUM_ROWS, NUM_COLS), dtype=int)

def valid_moves(board):
    """Return columns that are not full."""
    return np.where(board[0] == 0)[0]

def make_move(board, col, player):
    """Drop a player's piece in the chosen column."""
    for r in range(NUM_ROWS - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = player
            return

def check_win(board, player):
    """Check if the player has won the game."""
    # Horizontal
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS - 3):
            if np.all(board[r, c:c + 4] == player):
                return True
    # Vertical
    for c in range(NUM_COLS):
        for r in range(NUM_ROWS - 3):
            if np.all(board[r:r + 4, c] == player):
                return True
    # Diagonal TL-BR
    for r in range(NUM_ROWS - 3):
        for c in range(NUM_COLS - 3):
            if np.all([board[r+i, c+i] == player for i in range(4)]):
                return True
    # Diagonal BL-TR
    for r in range(3, NUM_ROWS):
        for c in range(NUM_COLS - 3):
            if np.all([board[r-i, c+i] == player for i in range(4)]):
                return True
    return False
