from constants import * 


def vertical_moves(pos1):
    for row in ROWS:
        pos2 = pos1[0] + str(row)
        if pos1 != pos2:
            yield pos1 + pos2


def horizontal_moves(pos1):
    # Castling is included in the horizontal movement.
    # e1c1, e1g1 are white castling moves and e8c8 and e8g8 are black castling moves.
    for col in COLS:
        pos2 = col + pos1[1]
        if pos1 != pos2:
            yield pos1 + pos2


def diagonal_moves(pos1):
    r, c = np.where(BOARD == pos1)
    r, c = r[0], c[0]
    
    row_shifter = col_shifter = range(-8, 8, 1)
    for k in [1, -1]:
        # k = 1 means top left to bottom right diagonal and 
        #    -1 means top right to bottom left diagonal
        for i, j in zip(row_shifter, col_shifter):
            rr = r + k * i
            cc = c + j
            if 0 <= rr < 8 and 0 <= cc < 8 and (r, c) != (rr, cc):
                yield pos1 + BOARD[rr, cc]
                

def knight_moves(pos1):
    r, c = np.where(BOARD == pos1)
    r, c = r[0], c[0]
    
    for k in range(2):
        for i in [-2, 2]:       # Two horizontal or vertical move
            for j in [-1, 1]:   # One vertical or horizontal move
                
                if k: rr, cc = r + i, c + j   # 2 horizontal then 1 vertical move
                else: rr, cc = r + j, c + i   # 1 horizontal then 2 vertical move

                if 0 <= rr < 8 and 0 <= cc < 8 and (r, c) != (rr, cc):
                    yield pos1 + BOARD[rr, cc]
    
    # for j in [-2, 2]:
    #     for i in [-1, 1]:
    #         rr, cc = r + i, c + j
    #         if 0 <= rr < 8 and 0 <= cc < 8 and (r, c) != (rr, cc):
    #             yield pos1 + BOARD[rr, cc]


def promotion_moves(pos1):
    if not pos1[1] in {'2', '7'}:
        return
    
    ri = COLS.index(pos1[0])
    pos2_col = '1' if pos1[1] == '2' else '8'
    for i in [-1, 0, 1]:
        col_index = ri + i
        if 0 <= col_index < 8:
            for piece in PIECES_TO_PROMOTE:
                for piece2 in [piece, piece.upper()]:   # Hack: piece type was creating problems. Fix this later.
                    # piece = piece.upper() if pos1[1] == '2' else piece
                    yield pos1 + COLS[col_index] + pos2_col + piece2
           
def get_all_legal_moves():      
    all_legal_moves = set()

    for pos1 in flat_board:
        unions_of_moves = set(vertical_moves(pos1)) | set(horizontal_moves(pos1)) | \
            set(diagonal_moves(pos1)) | set(knight_moves(pos1)) | set(promotion_moves(pos1))
            
        all_legal_moves.update(unions_of_moves)

    len(all_legal_moves)
    return dict(zip(all_legal_moves, range(len(all_legal_moves))))


ALL_LEGAL_MOVES = get_all_legal_moves()


if __name__ == "__main__":
    print(list(vertical_moves('a2')))
    print(list(horizontal_moves('a2')))
    print(list(diagonal_moves('a2')))
    print(list(knight_moves('c4'))) 
    print(list(promotion_moves('a2')))
    print(list(promotion_moves('g7')))