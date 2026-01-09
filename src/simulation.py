import numpy as np
from itertools import combinations
from src.board import create_board, valid_moves, make_move, check_win
from src.players import random_player, heuristic_player, intelligent_player

def simulate_game(player1_func, player2_func):
    """Simulate a single Connect-Four game."""
    board = create_board()
    turn = 1
    while True:
        move = player1_func(board, 1) if turn == 1 else player2_func(board, 2)
        make_move(board, move, turn)
        if check_win(board, turn):
            return turn
        if len(valid_moves(board)) == 0:
            return 0
        turn = 3 - turn

def monte_carlo(players_dict, base_sim=100):
    """Simulate all player pairings and compute probabilities, SE, and quotes."""
    results = {}
    player_names = list(players_dict.keys())

    for p1_name, p2_name in combinations(player_names, 2):
        sims = np.random.randint(100, 500)  # auto random simulation number
        func1, func2 = players_dict[p1_name], players_dict[p2_name]
        outcomes = {p1_name:0, p2_name:0, "Draw":0}

        for _ in range(sims):
            winner = simulate_game(func1, func2)
            if winner == 1:
                outcomes[p1_name] += 1
            elif winner == 2:
                outcomes[p2_name] += 1
            else:
                outcomes["Draw"] += 1

        prob_p1 = outcomes[p1_name] / sims
        prob_p2 = outcomes[p2_name] / sims
        prob_draw = outcomes["Draw"] / sims

        se_p1 = np.sqrt(prob_p1*(1-prob_p1)/sims)
        se_p2 = np.sqrt(prob_p2*(1-prob_p2)/sims)
        se_draw = np.sqrt(prob_draw*(1-prob_draw)/sims)

        quote_p1 = 1/prob_p1 if prob_p1>0 else np.inf
        quote_p2 = 1/prob_p2 if prob_p2>0 else np.inf
        quote_draw = 1/prob_draw if prob_draw>0 else np.inf

        results[(p1_name, p2_name)] = {
            "games": sims,
            "probabilities": (prob_p1, prob_p2, prob_draw),
            "SE": (se_p1, se_p2, se_draw),
            "quotes": (quote_p1, quote_p2, quote_draw),
            "outcomes": outcomes
        }
    return results