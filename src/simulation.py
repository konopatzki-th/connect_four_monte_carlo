from Board import create_board, make_move, check_win

def simulate_game(player1_func, player2_func,
                  moves_count_p1=None, moves_count_p2=None,
                  wins_count_p1=None, wins_count_p2=None):
    """Simulate a single Connect Four game and return the winner (1,2,0)."""
    board = create_board()
    turn = 1
    move_history_p1 = []
    move_history_p2 = []

    while True:
        if turn == 1:
            move = player1_func(board, turn)
            move_history_p1.append(move)
        else:
            move = player2_func(board, turn)
            move_history_p2.append(move)

        make_move(board, move, turn)

        if check_win(board, turn):
            # Optional: track wins per move (can leave None)
            return turn

        if len([c for c in range(7) if board[0, c] == 0]) == 0:
            return 0  # Draw

        turn = 3 - turn
