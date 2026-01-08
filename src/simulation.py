import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations

# =========================================================
# Configuration
# =========================================================

NUM_ROWS = 6  # Number of rows on the Connect Four board
NUM_COLS = 7  # Number of columns on the Connect Four board
BASE_SIM = 200  # Base number of simulations per player pair

# Ensure the "figures" directory exists for saving plots
os.makedirs("figures", exist_ok=True)

# Balance indicator: estimates relative skill differences between players
# Higher value → larger skill difference
balance_indicator = {
    ("4 Jahre", "7 Jahre"): 0.1,
    ("4 Jahre", "9 Jahre"): 0.1,
    ("4 Jahre", "11 Jahre"): 0.05,
    ("7 Jahre", "9 Jahre"): 0.5,
    ("7 Jahre", "11 Jahre"): 0.7,
    ("9 Jahre", "11 Jahre"): 0.8
}

# Adjust simulations per pair based on balance indicator
# Larger skill gaps → more simulations to better capture variability
simulations_per_pair = {
    pair: int(BASE_SIM * (1 + balance * 2))
    for pair, balance in balance_indicator.items()
}

# =========================================================
# Game Logic
# =========================================================

def create_board():
    """
    Create an empty Connect Four board.
    0 = empty, 1 = player 1, 2 = player 2
    """
    return np.zeros((NUM_ROWS, NUM_COLS), dtype=int)

def valid_moves(board):
    """
    Return a list of column indices where a piece can be dropped.
    Only columns where the top cell is empty are valid.
    """
    return np.where(board[0] == 0)[0]

def make_move(board, col, player):
    """
    Drop a piece for the given player into the specified column.
    The piece falls to the lowest empty row.
    """
    for r in range(NUM_ROWS - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = player
            return

def check_win(board, player):
    """
    Check if the given player has won the game.
    Win conditions: 4 in a row horizontally, vertically, or diagonally.
    """
    # Horizontal check
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS - 3):
            if np.all(board[r, c:c+4] == player):
                return True

    # Vertical check
    for c in range(NUM_COLS):
        for r in range(NUM_ROWS - 3):
            if np.all(board[r:r+4, c] == player):
                return True

    # Diagonal down-right check
    for r in range(NUM_ROWS - 3):
        for c in range(NUM_COLS - 3):
            if np.all([board[r+i, c+i] == player for i in range(4)]):
                return True

    # Diagonal up-right check
    for r in range(3, NUM_ROWS):
        for c in range(NUM_COLS - 3):
            if np.all([board[r-i, c+i] == player for i in range(4)]):
                return True

    return False

# =========================================================
# Player Strategies
# =========================================================

def random_player(board, _player):
    """
    Player that selects a valid move at random.
    """
    return np.random.choice(valid_moves(board))

def heuristic_player(board, player):
    """
    Simple AI:
    1. If a winning move is available, take it.
    2. Block opponent's winning move if possible.
    3. Prefer the center column.
    4. Otherwise, choose randomly.
    """
    opponent = 3 - player

    # Try to win
    for move in valid_moves(board):
        temp = board.copy()
        make_move(temp, move, player)
        if check_win(temp, player):
            return move

    # Block opponent
    for move in valid_moves(board):
        temp = board.copy()
        make_move(temp, move, opponent)
        if check_win(temp, opponent):
            return move

    # Prefer center column if available
    center = NUM_COLS // 2
    if center in valid_moves(board):
        return center

    # Otherwise pick a random valid move
    return np.random.choice(valid_moves(board))

def intelligent_player(board, player, depth=1):
    """
    Minimax-based AI:
    Evaluates all valid moves up to a certain depth to select the best move.
    """
    opponent = 3 - player
    best_score = -np.inf
    best_move = valid_moves(board)[0]

    for move in valid_moves(board):
        temp = board.copy()
        make_move(temp, move, player)
        score = minimax(temp, depth - 1, False, player, opponent)
        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def minimax(board, depth, maximizing, player, opponent):
    """
    Recursive minimax evaluation:
    - maximizing=True → player's turn
    - maximizing=False → opponent's turn
    Base cases:
        - Win/loss detected
        - Depth limit reached
        - No valid moves (draw)
    """
    if check_win(board, player):
        return 100  # player wins
    if check_win(board, opponent):
        return -100  # opponent wins
    if depth == 0 or len(valid_moves(board)) == 0:
        return evaluate_board(board, player)

    if maximizing:
        # Try all moves for player and select maximum score
        return max(
            minimax(apply_move(board, m, player), depth - 1, False, player, opponent)
            for m in valid_moves(board)
        )
    else:
        # Try all moves for opponent and select minimum score
        return min(
            minimax(apply_move(board, m, opponent), depth - 1, True, player, opponent)
            for m in valid_moves(board)
        )

def apply_move(board, move, player):
    """
    Helper to simulate a move without modifying the original board.
    Returns a new board state.
    """
    temp = board.copy()
    make_move(temp, move, player)
    return temp

def evaluate_board(board, player):
    """
    Simple evaluation function:
    - +10 for potential winning line (3 in a row + empty space)
    - -8 for opponent's potential winning line
    """
    score = 0
    opponent = 3 - player
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS - 3):
            window = board[r, c:c+4]
            if np.count_nonzero(window == player) == 3 and np.count_nonzero(window == 0) == 1:
                score += 10
            if np.count_nonzero(window == opponent) == 3 and np.count_nonzero(window == 0) == 1:
                score -= 8
    return score

# =========================================================
# Simulation
# =========================================================

def simulate_game(player1, player2):
    """
    Simulate a single game between player1 and player2.
    Returns:
        1 → player1 wins
        2 → player2 wins
        0 → draw
    """
    board = create_board()
    turn = 1

    while True:
        # Determine move based on current player's strategy
        if turn == 1:
            move = player1(board, 1)
        else:
            move = player2(board, 2)

        make_move(board, move, turn)

        # Check for win
        if check_win(board, turn):
            return turn

        # Check for draw
        if len(valid_moves(board)) == 0:
            return 0

        # Switch turn
        turn = 3 - turn

# =========================================================
# Main Execution
# =========================================================

if __name__ == "__main__":

    # Define player types (age-named for skill levels)
    players = {
        "4 Jahre": random_player,
        "7 Jahre": heuristic_player,
        "9 Jahre": lambda b, p: intelligent_player(b, p, depth=1),
        "11 Jahre": lambda b, p: intelligent_player(b, p, depth=2)
    }

    starting_advantages = {}  # Stores advantage metrics for all matchups

    # Iterate over all unique player pairs
    for p1, p2 in combinations(players.keys(), 2):
        SIMS = simulations_per_pair[(p1, p2)]  # Number of simulations for this pair

        outcomes = {p1: 0, p2: 0, "Draw": 0}  # Track wins and draws
        advantage_samples = []  # Store advantage values after each game

        for _ in range(SIMS):
            result = simulate_game(players[p1], players[p2])

            # Update outcome counters
            if result == 1:
                outcomes[p1] += 1
            elif result == 2:
                outcomes[p2] += 1
            else:
                outcomes["Draw"] += 1

            # Compute running advantage metric
            if (outcomes[p1] + outcomes[p2]) > 0:
                adv = (outcomes[p1] - outcomes[p2]) / (outcomes[p1] + outcomes[p2])
                advantage_samples.append(adv)

        # Compute overall starting player advantage for this matchup
        total_games = sum(outcomes.values())
        advantage = (outcomes[p1] - outcomes[p2]) / total_games
        starting_advantages[f"{p1} vs {p2}"] = advantage

        # ---------------------------------
        # FIGURE 1: Distribution of Monte Carlo advantage
        # ---------------------------------
        plt.figure(figsize=(7, 4))
        plt.hist(advantage_samples, bins=20)
        plt.xlabel("Starting Player Advantage")
        plt.ylabel("Frequency")
        plt.title(f"Advantage Distribution: {p1} vs {p2}")

        fname = f"figures/{p1}_vs_{p2}_advantage_distribution.png".replace(" ", "_")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()

    # ---------------------------------
    # FIGURE 2: Overview – Starting Player Advantage
    # ---------------------------------
    labels = list(starting_advantages.keys())
    values = list(starting_advantages.values())

    plt.figure(figsize=(10, 5))
    plt.bar(labels, values)
    plt.axhline(0, linestyle="--")  # Reference line at 0
    plt.ylabel("Win Probability Difference")
    plt.title("Starting Player Advantage Across All Matchups")
    plt.xticks(rotation=30, ha="right")

    plt.savefig("figures/starting_player_advantage_overview.png",
                dpi=300, bbox_inches="tight")
    plt.close()
