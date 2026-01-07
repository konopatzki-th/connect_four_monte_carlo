import numpy as np

ROWS = 6
COLS = 7

class ConnectFour:
    def __init__(self, start_player=1):
        self.board = np.zeros((ROWS, COLS))
        self.current_player = start_player
        self.winner = 0
        self.finished = False

    def get_valid_moves(self):
        return [c for c in range(COLS) if self.board[0][c] == 0]

    def make_move(self, col):
        for row in range(ROWS - 1, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                break

        if self.check_win(self.current_player):
            self.winner = self.current_player
            self.finished = True
        elif not self.get_valid_moves():
            self.finished = True

        self.current_player *= -1

    def check_win(self, player):
        for r in range(ROWS):
            for c in range(COLS - 3):
                if all(self.board[r][c+i] == player for i in range(4)):
                    return True

        for r in range(ROWS - 3):
            for c in range(COLS):
                if all(self.board[r+i][c] == player for i in range(4)):
                    return True

        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                if all(self.board[r+i][c+i] == player for i in range(4)):
                    return True

        for r in range(3, ROWS):
            for c in range(COLS - 3):
                if all(self.board[r-i][c+i] == player for i in range(4)):
                    return True

        return False
