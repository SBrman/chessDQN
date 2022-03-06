import numpy as np

# Constants
# Visualizing board
BOARD_SIZE = 500    # 500 * 500 pixel^2 board.

PIECES_TO_PROMOTE = 'qrnb'
COLS = 'abcdefgh'
ROWS = list(range(1, 9))
BOARD = np.array([[f'{col}{row}' for col in COLS] for row in ROWS[::-1]])
flat_board = BOARD.flatten()

# Piece maps    
PIECE_MAP = {'p': -1, 'r': -5, 'n': -4, 'b': -3, 'q': -9, 'k': -10, '.': 0}
PIECE_MAP = PIECE_MAP | {piece.upper(): piece_id * -1 for piece, piece_id in PIECE_MAP.items()}

# There are some illegal moves here, fix it later?
all_moves = {f'{m1}{m2}' for m1 in flat_board for m2 in flat_board if m1 != m2}