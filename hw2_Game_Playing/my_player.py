from copy import deepcopy
from math import inf, sqrt
import time, os.path
import numpy as np

SIZE = 5
KOMI = SIZE/2
BLACK = 1
WHITE = 2
MAX_DEPTH = 5

def write_turn(n):
    with open('turn_count.csv', 'w') as f:
        f.write(str(n))

def read_turn():
    with open('turn_count.csv', 'r') as f:
        lines = f.readlines()
        n = lines[0]
    return int(n)

def readInput(n, path="input.txt"):
    with open(path, 'r') as f:
        lines = f.readlines()

        piece_type = int(lines[0])

        previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
        board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]

        return piece_type, previous_board, board

def writeOutput(result, path="output.txt"):
    res = ""
    if result == "PASS":
        res = "PASS"
    else:
        res += str(result[0]) + ',' + str(result[1])

    with open(path, 'w') as f:
        f.write(res)

def writePass(path="output.txt"):
	with open(path, 'w') as f:
		f.write("PASS")

def compare_board(board1, board2):
    for i in range(5):
        for j in range(5):
            if board1[i][j] != board2[i][j]:
                return False
    return True

def copy_board(board):
    return deepcopy(board)

def detect_neighbor(i, j, board):
    '''
    Detect all the neighbors of a given stone.

    :param i: row number of the board.
    :param j: column number of the board.
    :return: a list containing the neighbors row and column (row, column) of position (i, j).
    '''
    neighbors = []
    # Detect borders and add neighbor coordinates
    if i > 0: neighbors.append((i-1, j))
    if i < len(board) - 1: neighbors.append((i+1, j))
    if j > 0: neighbors.append((i, j-1))
    if j < len(board) - 1: neighbors.append((i, j+1))
    return neighbors    

def detect_neighbor_ally(i, j, board):
    '''
    Detect the neighbor allies of a given stone.

    :param i: row number of the board.
    :param j: column number of the board.
    :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
    '''
    neighbors = detect_neighbor(i, j, board)  # Detect neighbors
    group_allies = []
    # Iterate through neighbors
    for piece in neighbors:
        # Add to allies list if having the same color
        if board[piece[0]][piece[1]] == board[i][j]:
            group_allies.append(piece)
    return group_allies

def ally_dfs(i, j, board):
    '''
    Using DFS to search for all allies of a given stone.

    :param i: row number of the board.
    :param j: column number of the board.
    :return: a list containing the all allies row and column (row, column) of position (i, j).
    '''
    stack = [(i, j)]  # stack for DFS serach
    ally_members = []  # record allies positions during the search
    while stack:
        piece = stack.pop()
        ally_members.append(piece)
        neighbor_allies = detect_neighbor_ally(piece[0], piece[1], board)
        for ally in neighbor_allies:
            if ally not in stack and ally not in ally_members:
                stack.append(ally)
    return ally_members

def find_liberty(i, j, board):
    '''
    Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

    :param i: row number of the board.
    :param j: column number of the board.
    :return: boolean indicating whether the given stone still has liberty.
    '''
    ally_members = ally_dfs(i, j, board)
    for member in ally_members:
        neighbors = detect_neighbor(member[0], member[1], board)
        for piece in neighbors:
            # If there is empty space around a piece, it has liberty
            if board[piece[0]][piece[1]] == 0:
                return True
    # If none of the pieces in a allied group has an empty space, it has no liberty
    return False

def find_died_pieces(piece_type, board):
    '''
    Find the died stones that has no liberty in the board for a given piece type.

    :param piece_type: 1('X') or 2('O').
    :return: a list containing the dead pieces row and column(row, column).
    '''
    died_pieces = []

    for i in range(len(board)):
        for j in range(len(board)):
            # Check if there is a piece at this position:
            if board[i][j] == piece_type:
                # The piece die if it has no liberty
                if not find_liberty(i, j, board):
                    died_pieces.append((i,j))
    # print(f"died pieces {died_pieces}")
    return died_pieces

def remove_died_pieces(piece_type, board):
    '''
    Remove the dead stones in the board.

    :param piece_type: 1('X') or 2('O').
    :return: locations of dead pieces.
    '''

    died_pieces = find_died_pieces(piece_type, board)
    if not died_pieces: return []
    remove_certain_pieces(died_pieces, board)
    return died_pieces

def remove_certain_pieces(positions, board): 
    '''
    Remove the stones of certain locations.

    :param positions: a list containing the pieces to be removed row and column(row, column)
    :return: None.
    '''
    for piece in positions:
        board[piece[0]][piece[1]] = 0

def place_chess(i, j, piece_type, board, previous_board):
    '''
    Place a chess stone in the board.

    :param i: row number of the board.
    :param j: column number of the board.
    :param piece_type: 1('X') or 2('O').
    :return: boolean indicating whether the placement is valid.
    '''
    valid_place = valid_place_check(i, j, piece_type, board, previous_board)
    if not valid_place:
        return False
    
    board[i][j] = piece_type
    return True

def valid_place_check(i, j, piece_type, board, previous_board, test_check=False, verbose=False):
    '''
    Check whether a placement is valid.

    :param i: row number of the board.
    :param j: column number of the board.
    :param piece_type: 1(white piece) or 2(black piece).
    :param test_check: boolean if it's a test check.
    :return: boolean indicating whether the placement is valid.
    '''   
    if test_check:
        verbose = False

    # Check if the place is in the board range
    if not (i >= 0 and i < len(board)):
        if verbose:
            print(('Invalid placement. row should be in the range 1 to {}.').format(len(board) - 1))
        return False
    if not (j >= 0 and j < len(board)):
        if verbose:
            print(('Invalid placement. column should be in the range 1 to {}.').format(len(board) - 1))
        return False
    
    # Check if the place already has a piece
    if board[i][j] != 0:
        if verbose:
            print('Invalid placement. There is already a chess in this position.')
        return False
    
    # Copy the board for testing
    test_board = copy_board(board)

    # Check if the place has liberty
    test_board[i][j] = piece_type
    # test_go.update_board(test_board)
    if find_liberty(i, j, test_board):
        return True

    # If not, remove the died pieces of opponent and check again
    died_pieces = remove_died_pieces(3 - piece_type, test_board)
    if not find_liberty(i, j, test_board):
        if verbose:
            print('Invalid placement. No liberty found in this position.')
        return False

    # Check special case: repeat placement causing the repeat board state (KO rule)
    else:
        if died_pieces and compare_board(previous_board, test_board):
            if verbose:
                print('Invalid placement. A repeat move not permitted by the KO rule.')
            return False
    return True

def visualize_board(board):
    '''
    Visualize the board.

    :return: None
    '''
    print('-' * len(board) * 2)
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 0:
                print(' ', end=' ')
            elif board[i][j] == 1:
                print('X', end=' ')
            else:
                print('O', end=' ')
        print()
    print('-' * len(board) * 2)

def single_liberty_check(board, i, j, color):
    """ Determines whether or not the position i,j is a liberty
        color: color of the stone
        returns 0 or 1 """
    if board[i][j] == 0:
        if j-1 >= 0 and board[i][j-1] == color: #up
            return 1
        elif j+1 < SIZE and board[i][j+1] == color: #down
            return 1
        elif i-1 >= 0 and board[i-1][j] == color: #left
            return 1
        elif i+1 < SIZE and board[i+1][j] == color: #right
            return 1
    return 0

def future_boards(stone, board, previous_board):
    """ Generates all valid future states
        stone: generate moves for this color of stone
        returns list of tuples: (move, board)
    """
    future_boards = []
    for i in range(5):
        for j in range(5):
            if valid_place_check(i, j, stone, board, previous_board):                  
                new_board = future_board(i, j, stone, board, previous_board)
                if new_board != None:
                    future_boards.append(((i,j), new_board))
                    # future_boards.append((self.evaluate_heuristic(new_board.board), new_board, (i,j)))
    # if stone == BLACK: future_boards.sort(key=lambda h: h[0], reverse=True)
    # if stone == WHITE: future_boards.sort(key=lambda h: h[0])
    return future_boards

def future_board(i, j, stone, board, previous_board):
    copy = copy_board(board)
    if not place_chess(i, j, stone, copy, previous_board):
        return None
    remove_died_pieces(3-stone, copy)
    return copy

def distance(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def manhattan_distance(a, b):
    return np.abs(a[0]-b[0]) + np.abs(a[1]-b[1])

class Player:
    def __init__(self, stone, turn):
        self.player = stone
        self.turn = turn
        self.opponent = 3 - stone
        self.branching = 8
        self.manhattan_table = [[4,3,2,3,4],[3,2,1,2,3],[2,1,0,1,2],[3,2,1,2,3],[4,3,2,3,4]]
        self.nodes = 0
        self.pruned = 0
        self.transposition_table = {}
    
    # Utility functions =============================================
    def evaluate_heuristic(self, board):
        """ Evaluates the board
            returns heuristic value """
        my_stones, opp_stones = 0, 0
        my_liberties, opp_liberties = 0, 0
        my_group, opp_group = 0, 0
        my_distance, opp_distance = 0, 0
        for i in range(SIZE):
            for j in range(SIZE):
                if board[i][j] == self.player:
                    my_stones += 1
                    my_group = len(ally_dfs(i, j, board))
                    my_distance = self.manhattan_table[i][j]
                elif board[i][j] == self.opponent:
                    opp_stones += 1
                    opp_group = len(ally_dfs(i, j, board))
                    opp_distance = self.manhattan_table[i][j]
                else:
                    my_liberties += single_liberty_check(board, i, j, self.player)
                    opp_liberties += single_liberty_check(board, i, j, self.opponent)
        stones = my_stones - opp_stones
        liberties = my_liberties - opp_liberties
        groups = my_group - opp_group
        dist = opp_distance - my_distance
        return stones + 0.8 * liberties + groups + dist

    def order_moves(self, moves, reverse=True):
        """ Returns re-ordered moves based on heuristic value
        """
        order = []
        for i, move in enumerate(moves): # change filtered_moves back to moves to undo manhattan filter
            order.append((self.evaluate_heuristic(move[1]), i))
        order.sort(reverse=reverse)
        sorted_moves = []
        for i, o in enumerate(order):
            sorted_moves.append(moves[o[1]])
        return sorted_moves[:self.branching]
        
    def encode_board(self, board):
        return ''.join([str(board[i][j]) for i in range(5) for j in range(5)])

    def table_put(self, board, depth):
        encoded_board = self.encode_board(board)
        self.transposition_table[encoded_board] = (self.evaluate_heuristic(board), depth)

    def table_get(self, board):
        pass

    def negative(self, encoded_board):
        negative = list(encoded_board)
        for i, s in enumerate(negative):
            if s == "1":
                negative[i] = "2"
            elif s == "2":
                negative[i] = "1"
        return ''.join(negative)

    def get_equivalent(self, board):
        sym = set()
        sym.add(self.encode_board(board))
        sym.add(self.negative(self.encode_board(board)))
        for i in range(1,4):
            rot = np.rot90(board,i)
            flip = np.flip(rot)
            sym.add(self.encode_board(rot))
            sym.add(self.encode_board(flip))
            sym.add(self.negative(self.encode_board(rot)))
            sym.add(self.negative(self.encode_board(flip)))
        return sym

    # Player behavior ===============================================
    # Standard Minimax -------------------------------------------
    def minimax_decision(self, board, prev_board, stone):
        self.nodes += 1
        moves = future_boards(stone,board,prev_board)
        max_val = -inf
        best = None
        for move in moves:
            val = self.get_min(move[1],prev_board,self.opponent,1)
            if val > max_val:
                max_val = val
                best = move[0]
        return best

    def get_min(self, board, prev_board, stone, depth):
        self.nodes += 1
        if depth > MAX_DEPTH:
            return self.evaluate_heuristic(board)
        depth += 1
        moves = future_boards(stone, board, prev_board)
        min_val = inf
        for move in moves:
            val = self.get_max(move[1],board, self.player, depth)
            if val < min_val:
                min_val = val
        # print(f"min depth: {depth}")
        return min_val

    def get_max(self, board, prev_board, stone, depth):
        self.nodes += 1
        if depth > MAX_DEPTH:
            return self.evaluate_heuristic(board)
        depth += 1
        moves = future_boards(stone, board, prev_board)
        max_val = -inf
        for move in moves:
            val = self.get_min(move[1],board, self.opponent,depth)
            if val > max_val:
                max_val = val
        # print(f"max depth: {depth}")
        return max_val

    # Alpha Beta -------------------------------------------------
    def alpha_beta_decision(self, board, prev_board, stone):
        moves = future_boards(stone,board,prev_board)
        max_val = -inf
        best = None
        self.nodes += 1

        # Move ordering
        moves = self.order_moves(moves)

        filtered_moves = []
        # focus based on turn
        for move in moves:
            if self.manhattan_table[move[0][0]][move[0][1]] <= self.turn+1:
                filtered_moves.append(move)

        for move in filtered_moves:
            val = self.alpha_beta_min(move[1],prev_board,self.opponent,1, -inf, inf)
            if val > max_val:
                max_val = val
                best = move[0]
        return best

    def alpha_beta_min(self, board, prev_board, stone, depth, alpha, beta):
        self.nodes += 1
        if depth > MAX_DEPTH:
            return self.evaluate_heuristic(board)
        depth += 1
        moves = future_boards(stone, board, prev_board)
        min_val = inf
        # Move ordering
        moves = self.order_moves(moves, False)
        for move in moves:
            val = self.alpha_beta_max(move[1],board, self.player, depth, alpha, beta)
            min_val = min(min_val, val)
            if min_val <= alpha: 
                self.pruned += 1
                return min_val
            beta = min(beta, val)
        # print(f"min depth: {depth}")
        return min_val

    def alpha_beta_max(self, board, prev_board, stone, depth, alpha, beta):
        self.nodes += 1
        if depth > MAX_DEPTH:
            return self.evaluate_heuristic(board)
        depth += 1
        moves = future_boards(stone, board, prev_board)
        max_val = -inf
        # Move ordering
        moves = self.order_moves(moves)
        for move in moves:
            val = self.alpha_beta_min(move[1],board, self.opponent,depth, alpha, beta)
            max_val = max(max_val, val)
            if max_val >= beta: 
                self.pruned += 1
                return max_val
            alpha = max(alpha, max_val)
        # print(f"max depth: {depth}")
        return max_val


    # Greedy -----------------------------------------------------
    def greedy(self, board, prev_board, stone):
        moves = future_boards(stone, board, prev_board)
        max_val = -inf
        best = None
        for move in moves:
            val = self.evaluate_heuristic(move[1])
            if val > max_val:
                max_val = val
                best = move[0]
        return best

def main():
    begin = time.time()
    stone, previous_board, board = readInput(5)
    
    n = 0
    # Trying to figure out how to keep track of the turn count
    if np.sum(previous_board) == 0:
        n = 1
        write_turn(1)
    else:
        n = read_turn()
        write_turn(n+1)

    # check for beginning of game
    
    if np.sum(board) == 0:
        write_turn(n+1)
        writeOutput((2,2))
        return
    
    board_copy = copy_board(board)
    player = Player(stone, n)
    best = player.alpha_beta_decision(board, previous_board, stone)
    if best != None:
        place_chess(best[0],best[1],stone, board, previous_board)
        remove_died_pieces(3 - stone, board)

    if board_copy == board:
        write_turn(n+1)
        writePass()
    else:
        write_turn(n+1)
        writeOutput(best)

    print(f"Total nodes: {player.nodes}")
    print(f"Nodes prunes: {player.pruned}")
    print(f"Turn: {n+1}")
    print(f"Total time: {time.time() - begin}")



if __name__ == '__main__':
    main()