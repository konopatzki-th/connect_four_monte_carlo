import random
from game import ConnectFour

def simulate_games(n_games, start_player):
    results = {
        "start_player_wins": 0,
        "other_player_wins": 0,
        "draws": 0
    }

    for _ in range(n_games):
        game = ConnectFour(start_player=start_player)

        while not game.finished:
            move = random.choice(game.get_valid_moves())
            game.make_move(move)

        if game.winner == start_player:
            results["start_player_wins"] += 1
        elif game.winner == 0:
            results["draws"] += 1
        else:
            results["other_player_wins"] += 1

    return results
