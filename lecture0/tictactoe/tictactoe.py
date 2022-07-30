"""
Tic Tac Toe Player
"""

import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if board == initial_state():
        return X

    x_count, o_count = 0, 0
    for row in board:
        for cell in row:
            if cell == X:
                x_count += 1
            elif cell == O:
                o_count += 1

    return X if x_count == o_count else O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    actions = set()

    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == EMPTY:
                actions.add((i, j))

    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    (i, j) = action
    if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]):
        raise IndexError("Invalid action")

    new_board = [[cell for cell in row] for row in board]
    new_board[i][j] = player(board)

    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if check_rows(board, X) or check_columns(board, X) or check_diagonals(board, X):
        return X
    elif check_rows(board, O) or check_columns(board, O) or check_diagonals(board, O):
        return O


def check_rows(board, player):
    """
    Returns True if there is a full row of player's pieces.
    """
    for i in range(len(board)):
        if all(board[i][j] == player for j in range(len(board[0]))):
            return True

    return False


def check_columns(board, player):
    """
    Returns True if there is a full column of player's pieces.
    """
    for j in range(len(board[0])):
        if all(board[i][j] == player for i in range(len(board))):
            return True

    return False


def check_top_bottom_diagonal(board, player):
    """
    Returns True if there is a full top-bottom diagonal of player's pieces.
    """
    check = True

    for i in range(len(board)):
        if board[i][i] != player:
            check = False

    return check


def check_bottom_top_diagonal(board, player):
    """
    Returns True if there is a full bottom-top diagonal of player's pieces.
    """
    check = True

    for i in range(len(board)):
        if board[i][len(board) - 1 - i] != player:
            check = False

    return check


def check_diagonals(board, player):
    """
    Returns True if there is a full top-bottom diagonal of player's pieces.
    """
    return check_top_bottom_diagonal(board, player) or check_bottom_top_diagonal(board, player)


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) or is_tie(board):
        return True

    return False


def is_tie(board):
    """
    Returns True if the game is a tie, False otherwise.
    """
    if all(cell != EMPTY for row in board for cell in row):
        return True

    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1

    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    values = []
    if player(board) == X:
        for action in actions(board):
            values.append([min_value(result(board, action)), action])

        return sorted(values, reverse=True)[0][1]

    else:
        for action in actions(board):
            values.append([max_value(result(board, action)), action])

        return sorted(values)[0][1]


def min_value(board):
    """
    Returns the minimum value for the current player on the board.
    """
    if terminal(board):
        return utility(board)

    v = math.inf
    for action in actions(board):
        v = min(v, max_value(result(board, action)))

    return v


def max_value(board):
    """
    Returns the maximum value for the current player on the board.
    """
    if terminal(board):
        return utility(board)

    v = -math.inf
    for action in actions(board):
        v = max(v, min_value(result(board, action)))

    return v
