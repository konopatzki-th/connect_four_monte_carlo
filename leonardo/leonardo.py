import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# board 6x7
NUM_ROWS = 6
NUM_COLS = 7

BASE_SIM = 100

# Balance-Indikator (0=einseitig, 1=ausgeglichen)
balance_indicator = {
    ("4 Jahre","7 Jahre"): 0.1,
    ("4 Jahre","9 Jahre"): 0.1,
    ("4 Jahre","11 Jahre"): 0.05,
    ("7 Jahre","9 Jahre"): 0.5,
    ("7 Jahre","11 Jahre"): 0.7,
    ("9 Jahre","11 Jahre"): 0.8
}

# Berechne Anzahl Simulationen pro Paarung
simulations_per_pair = {}
for pair, balance in balance_indicator.items():
    simulations_per_pair[pair] = int(BASE_SIM * (1 + balance*2))



# board function


def create_board():
    return np.zeros((NUM_ROWS, NUM_COLS), dtype=int)
    # creates a 6x7 array with zeros


def valid_moves(board):
    return np.where(board[0] == 0)[0]
    # check possible moves


def make_move(board, col, player):
    for r in range(NUM_ROWS - 1, -1, -1):
        if board[r, col] == 0:
            board[r, col] = player
            return
# insert chip into board

def check_win(board, player): # check winner
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS - 3):
            if np.all(board[r, c:c + 4] == player):
                return True
            # check horizontally
    for c in range(NUM_COLS):
        for r in range(NUM_ROWS - 3):
            if np.all(board[r:r + 4, c] == player):
                return True
            # check vertically
    for r in range(NUM_ROWS - 3):
        for c in range(NUM_COLS - 3):
            if np.all([board[r + delta, c + delta] == player for delta in range(4)]):
                return True
            # check diagonally from top to bottom
    for r in range(3, NUM_ROWS):
        for c in range(NUM_COLS - 3):
            if np.all([board[r - delta, c + delta] == player for delta in range(4)]):
                return True
            # Check diagonally from bottom to top
    return False # no win


# player function


def random_player(board, _player): # _player is not required here, but is included for interface compatibility.
    return np.random.choice(valid_moves(board))


def heuristic_player(board, player_id):
    opponent = 3 - player_id
    moves = valid_moves(board)
    for move in moves:
        temp = board.copy()
        make_move(temp, move, player_id)
        if check_win(temp, player_id):
            return move
    for move in moves:
        temp = board.copy()
        make_move(temp, move, opponent)
        if check_win(temp, opponent):
            return move
    center = NUM_COLS // 2
    if center in moves:
        return center
    return np.random.choice(moves)


def intelligent_player(board, player_id, depth=1):
    opponent = 3 - player_id
    moves = valid_moves(board)
    best_score = -np.inf
    best_move = moves[0]
    for move in moves:
        temp = board.copy()
        make_move(temp, move, player_id)
        score = minimax(temp, depth - 1, False, player_id, opponent)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


def minimax(board, depth, maximizing, player_id, opponent):
    if check_win(board, player_id): return 100
    if check_win(board, opponent): return -100
    if len(valid_moves(board)) == 0 or depth == 0:
        return evaluate_board(board, player_id)
    if maximizing:
        max_eval = -np.inf
        for move in valid_moves(board):
            temp = board.copy()
            make_move(temp, move, player_id)
            move_score = minimax(temp, depth - 1, False, player_id, opponent)
            max_eval = max(max_eval, move_score)
        return max_eval
    else:
        min_eval = np.inf
        for move in valid_moves(board):
            temp = board.copy()
            make_move(temp, move, opponent)
            move_score = minimax(temp, depth - 1, True, player_id, opponent)
            min_eval = min(min_eval, move_score)
        return min_eval


def evaluate_board(board, player):
    score = 0
    opponent = 3 - player
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS - 3):
            line = board[r, c:c + 4]
            if np.count_nonzero(line == player) == 3 and np.count_nonzero(line == 0) == 1:
                score += 10
            if np.count_nonzero(line == player) == 2 and np.count_nonzero(line == 0) == 2:
                score += 1
            if np.count_nonzero(line == opponent) == 3 and np.count_nonzero(line == 0) == 1:
                score -= 8
    return score


# -------------------------
# Simulation mit Spalten + Siege
# -------------------------
def simulate_game_with_wins(player1_func, player2_func, moves_count_p1, moves_count_p2, wins_count_p1, wins_count_p2):
    board = create_board()
    turn = 1
    move_history_p1, move_history_p2 = [], []

    while True:
        if turn == 1:
            move = player1_func(board, turn)
            moves_count_p1[move] += 1
            move_history_p1.append(move)
        else:
            move = player2_func(board, turn)
            moves_count_p2[move] += 1
            move_history_p2.append(move)
        make_move(board, move, turn)
        if check_win(board, turn):
            if turn == 1:
                for m in move_history_p1: wins_count_p1[m] += 1
            else:
                for m in move_history_p2: wins_count_p2[m] += 1
            return turn
        if len(valid_moves(board)) == 0:
            return 0
        turn = 3 - turn


# -------------------------
# Spieler + Alterslabel
# -------------------------
players = {
    "4 Jahre": random_player,
    "7 Jahre": heuristic_player,
    "9 Jahre": lambda b, p: intelligent_player(b, p, depth=1),
    "11 Jahre": lambda b, p: intelligent_player(b, p, depth=2)
}

# -------------------------
# Simulation und Quotenberechnung
# -------------------------
results = {}

for p1_name, p2_name in combinations(players.keys(), 2):
    # Anzahl Simulationen für diese Paarung
    SIMULATIONEN = simulations_per_pair[(p1_name, p2_name)]
    func1 = players[p1_name]
    func2 = players[p2_name]
    outcomes = {p1_name:0, p2_name:0, "Unentschieden":0}
    columns_p1 = np.zeros(NUM_COLS, dtype=int)
    columns_p2 = np.zeros(NUM_COLS, dtype=int)
    wins_p1 = np.zeros(NUM_COLS, dtype=int)
    wins_p2 = np.zeros(NUM_COLS, dtype=int)

    for i in range(SIMULATIONEN):
        winner = simulate_game_with_wins(func1, func2, columns_p1, columns_p2, wins_p1, wins_p2)
        if winner == 1:
            outcomes[p1_name] += 1
        elif winner == 2:
            outcomes[p2_name] += 1
        else:
            outcomes["Unentschieden"] += 1
        if (i + 1) % 20 == 0:
            print(f"{p1_name} vs {p2_name}: {i + 1}/{SIMULATIONEN} Spiele abgeschlossen")

    # Wahrscheinlichkeiten berechnen
    prob_p1 = float(outcomes[p1_name]) / SIMULATIONEN
    prob_p2 = float(outcomes[p2_name]) / SIMULATIONEN
    prob_draw = float(outcomes["Unentschieden"]) / SIMULATIONEN
    # Standardfehler berechnen
    se_p1 = np.sqrt(prob_p1 * (1 - prob_p1) / SIMULATIONEN)
    se_p2 = np.sqrt(prob_p2 * (1 - prob_p2) / SIMULATIONEN)
    se_draw = np.sqrt(prob_draw * (1 - prob_draw) / SIMULATIONEN)
    print(f"{p1_name}: Wahrscheinlichkeit={prob_p1:.3f}, SE={se_p1:.3f}")
    print(f"{p2_name}: Wahrscheinlichkeit={prob_p2:.3f}, SE={se_p2:.3f}")
    print(f"Unentschieden: Wahrscheinlichkeit={prob_draw:.3f}, SE={se_draw:.3f}")

    # Quoten berechnen (vereinfacht: Quote = 1 / Wahrscheinlichkeit)
    quote_p1 = (1 / prob_p1 if prob_p1 > 0 else np.inf)
    quote_p2 = (1 / prob_p2 if prob_p2 > 0 else np.inf)
    quote_draw = (1 / prob_draw if prob_draw > 0 else np.inf)

    results[(p1_name, p2_name)] = {
        "outcomes": outcomes,
        "probabilities": (prob_p1, prob_p2, prob_draw),
        "quotes": (quote_p1, quote_p2, quote_draw),
        "columns_p1": columns_p1,
        "columns_p2": columns_p2,
        "wins_p1": wins_p1,
        "wins_p2": wins_p2
    }

# -------------------------
# Ergebnisse ausgeben
# -------------------------
for (p1, p2), data in results.items():
    print(f"\n{p1} vs {p2} ({SIMULATIONEN} Spiele):")
    print(
        f"{p1} gewinnt: {data['outcomes'][p1]} ({data['probabilities'][0] * 100:.1f}%) → Quote ≈ {data['quotes'][0]:.2f}")
    print(
        f"{p2} gewinnt: {data['outcomes'][p2]} ({data['probabilities'][1] * 100:.1f}%) → Quote ≈ {data['quotes'][1]:.2f}")
    print(
        f"Unentschieden: {data['outcomes']['Unentschieden']} ({data['probabilities'][2] * 100:.1f}%) → Quote ≈ {data['quotes'][2]:.2f}")

    # Heatmap: Siege pro Spalte
    x = np.arange(NUM_COLS)
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, data["wins_p1"], width, label=f"{p1} Siege", color='blue')
    plt.bar(x + width / 2, data["wins_p2"], width, label=f"{p2} Siege", color='red')
    plt.xlabel("Spalte")
    plt.ylabel("Anzahl der Siege")
    plt.title(f"Siegspalten: {p1} vs {p2}")
    plt.xticks(x)
    plt.legend()
    plt.show()
