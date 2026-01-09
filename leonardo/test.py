# =========================================================
# Connect Four Monte-Carlo Simulation with automatic balance
# =========================================================
import numpy as np
from itertools import combinations

# -------------------------
# Board setup
# -------------------------
NUM_ROWS = 6
NUM_COLS = 7

def create_board():
    return np.zeros((NUM_ROWS, NUM_COLS), dtype=int)

def valid_moves(board):
    return np.where(board[0]==0)[0]

def make_move(board, col, player):
    for r in range(NUM_ROWS-1, -1, -1):
        if board[r, col]==0:
            board[r, col] = player
            return

def check_win(board, player):
    # Horizontal
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS-3):
            if np.all(board[r,c:c+4]==player):
                return True
    # Vertical
    for c in range(NUM_COLS):
        for r in range(NUM_ROWS-3):
            if np.all(board[r:r+4,c]==player):
                return True
    # Diagonal TL-BR
    for r in range(NUM_ROWS-3):
        for c in range(NUM_COLS-3):
            if all(board[r+i,c+i]==player for i in range(4)):
                return True
    # Diagonal BL-TR
    for r in range(3, NUM_ROWS):
        for c in range(NUM_COLS-3):
            if all(board[r-i,c+i]==player for i in range(4)):
                return True
    return False

# -------------------------
# Player definitions
# -------------------------
def random_player(board, _):
    return np.random.choice(valid_moves(board))

def heuristic_player(board, player):
    opponent = 3 - player
    moves = valid_moves(board)
    for m in moves:  # try to win
        temp = board.copy()
        make_move(temp, m, player)
        if check_win(temp, player): return m
    for m in moves:  # block opponent
        temp = board.copy()
        make_move(temp, m, opponent)
        if check_win(temp, opponent): return m
    center = NUM_COLS//2
    if center in moves: return center
    return np.random.choice(moves)

def intelligent_player(board, player, depth=1):
    opponent = 3 - player
    best_score = -np.inf
    best_move = valid_moves(board)[0]
    for m in valid_moves(board):
        temp = board.copy()
        make_move(temp, m, player)
        score = minimax(temp, depth-1, False, player, opponent)
        if score>best_score:
            best_score = score
            best_move = m
    return best_move

def minimax(board, depth, maximizing, player, opponent):
    if check_win(board, player): return 100
    if check_win(board, opponent): return -100
    if depth==0 or len(valid_moves(board))==0:
        return evaluate_board(board, player)
    if maximizing:
        return max(minimax(apply_move(board, m, player), depth-1, False, player, opponent)
                   for m in valid_moves(board))
    else:
        return min(minimax(apply_move(board, m, opponent), depth-1, True, player, opponent)
                   for m in valid_moves(board))

def apply_move(board, col, player):
    new_board = board.copy()
    make_move(new_board, col, player)
    return new_board

def evaluate_board(board, player):
    score = 0
    opponent = 3 - player
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS-3):
            window = board[r,c:c+4]
            if np.count_nonzero(window==player)==3 and np.count_nonzero(window==0)==1: score +=10
            if np.count_nonzero(window==player)==2 and np.count_nonzero(window==0)==2: score +=1
            if np.count_nonzero(window==opponent)==3 and np.count_nonzero(window==0)==1: score -=8
    return score

# -------------------------
# Simulate a single game with tracking (used for pre-sim and main sim)
# -------------------------
def simulate_game(player1_func, player2_func):
    board = create_board()
    turn = 1
    while True:
        move = player1_func(board, 1) if turn==1 else player2_func(board, 2)
        make_move(board, move, turn)
        if check_win(board, turn): return turn
        if len(valid_moves(board))==0: return 0
        turn = 3 - turn

# -------------------------
# Players setup
# -------------------------
players = {
    "4 Years": random_player,
    "7 Years": heuristic_player,
    "9 Years": lambda b,p: intelligent_player(b,p,depth=1),
    "11 Years": lambda b,p: intelligent_player(b,p,depth=2)
}

NUM_PRE_SIM = 20  # quick pre-simulation
SIM_RANGE = (100,500)  # min and max simulations

# -------------------------
# Main Monte Carlo loop
# -------------------------
results = {}

for p1_name, p2_name in combinations(players.keys(), 2):
    func1 = players[p1_name]
    func2 = players[p2_name]

    # --- Pre-simulation to determine balance
    wins_pre = [0,0,0]
    for _ in range(NUM_PRE_SIM):
        w = simulate_game(func1, func2)
        if w==1: wins_pre[0]+=1
        elif w==2: wins_pre[1]+=1
        else: wins_pre[2]+=1

    total_pre = sum(wins_pre)
    prob1 = wins_pre[0]/total_pre
    prob2 = wins_pre[1]/total_pre
    balance = 1 - abs(prob1 - prob2)
    SIMULATIONEN = int(SIM_RANGE[0] + (SIM_RANGE[1]-SIM_RANGE[0])*balance)
    print(f"{p1_name} vs {p2_name}: pre-sim balance={balance:.2f}, sims={SIMULATIONEN}")

    # --- Full simulation
    outcomes = {p1_name:0, p2_name:0, "Draw":0}
    for _ in range(SIMULATIONEN):
        w = simulate_game(func1, func2)
        if w==1: outcomes[p1_name]+=1
        elif w==2: outcomes[p2_name]+=1
        else: outcomes["Draw"]+=1

    # --- Probabilities & standard errors
    prob_p1 = outcomes[p1_name]/SIMULATIONEN
    prob_p2 = outcomes[p2_name]/SIMULATIONEN
    prob_draw = outcomes["Draw"]/SIMULATIONEN

    se_p1 = np.sqrt(prob_p1*(1-prob_p1)/SIMULATIONEN)
    se_p2 = np.sqrt(prob_p2*(1-prob_p2)/SIMULATIONEN)
    se_draw = np.sqrt(prob_draw*(1-prob_draw)/SIMULATIONEN)

    # --- Quotes
    quote_p1 = 1/prob_p1 if prob_p1>0 else np.inf
    quote_p2 = 1/prob_p2 if prob_p2>0 else np.inf
    quote_draw = 1/prob_draw if prob_draw>0 else np.inf

    results[(p1_name,p2_name)] = {
        "outcomes": outcomes,
        "probabilities": (prob_p1, prob_p2, prob_draw),
        "se": (se_p1, se_p2, se_draw),
        "quotes": (quote_p1, quote_p2, quote_draw)
    }

# -------------------------
# Print results
# -------------------------
for (p1,p2), data in results.items():
    print(f"\n{p1} vs {p2} ({sum(data['outcomes'].values())} games)")
    print(f"{p1}: {data['probabilities'][0]*100:.1f}% ± {data['se'][0]*100:.1f}% → Quote ≈ {data['quotes'][0]:.2f}")
    print(f"{p2}: {data['probabilities'][1]*100:.1f}% ± {data['se'][1]*100:.1f}% → Quote ≈ {data['quotes'][1]:.2f}")
    print(f"Draw: {data['probabilities'][2]*100:.1f}% ± {data['se'][2]*100:.1f}% → Quote ≈ {data['quotes'][2]:.2f}")
