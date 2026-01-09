from board import create_board, make_move, check_win

def simulate_game(player1_func, player2_func):
    board = create_board()
    turn = 1
    while True:
        move = player1_func(board, 1) if turn == 1 else player2_func(board, 2)
        make_move(board, move, turn)
        if check_win(board, turn):
            return turn
        if len([c for c in range(7) if board[0, c]==0]) == 0:
            return 0
        turn = 3 - turn
