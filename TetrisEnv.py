import random
import cv2
import numpy as np
from PIL import Image
from time import sleep

class TetrisBlocks():
    LINEBLOCK = { # Light Blue
        0: [(0,0), (1,0), (2,0), (3,0)],
        90: [(1,0), (1,1), (1,2), (1,3)],
        180: [(3,0), (2,0), (1,0), (0,0)],
        270: [(1,3), (1,2), (1,1), (1,0)],
    }
    TBLOCK = { # Purple
        0: [(1,0), (0,1), (1,1), (2,1)],
        90: [(0,1), (1,2), (1,1), (1,0)],
        180: [(1,2), (2,1), (1,1), (0,1)],
        270: [(2,1), (1,0), (1,1), (1,2)],
    }
    FORWARDLBLOCK = { # Orange
        0: [(1,0), (1,1), (1,2), (2,2)],
        90: [(0,1), (1,1), (2,1), (2,0)],
        180: [(1,2), (1,1), (1,0), (0,0)],
        270: [(2,1), (1,1), (0,1), (0,2)],
    }
    BACKWARDLBLOCK = { # Dark Blue
        0: [(1,0), (1,1), (1,2), (0,2)],
        90: [(0,1), (1,1), (2,1), (2,2)],
        180: [(1,2), (1,1), (1,0), (2,0)],
        270: [(2,1), (1,1), (0,1), (0,0)],
    }
    BACKWARDZBLOCK = { # Green
        0: [(0,0), (1,0), (1,1), (2,1)],
        90: [(0,2), (0,1), (1,1), (1,0)],
        180: [(2,1), (1,1), (1,0), (0,0)],
        270: [(1,0), (1,1), (0,1), (0,2)],
    }
    FORWARDZBLOCK = { # Red
        0: [(2,0), (1,0), (1,1), (0,1)],
        90: [(0,0), (0,1), (1,1), (1,2)],
        180: [(0,1), (1,1), (1,0), (2,0)],
        270: [(1,2), (1,1), (0,1), (0,0)],
    }
    SQUAREBLOCK = { # Yellow
        0: [(1,0), (2,0), (1,1), (2,1)],
        90: [(1,0), (2,0), (1,1), (2,1)],
        180: [(1,0), (2,0), (1,1), (2,1)],
        270: [(1,0), (2,0), (1,1), (2,1)],
    }
    DEFAULTBLOCKS = [LINEBLOCK, TBLOCK, FORWARDLBLOCK, BACKWARDLBLOCK, BACKWARDZBLOCK, FORWARDZBLOCK, SQUAREBLOCK]
    SINGLEBLOCK = {
        0: [(0,0)],
        90: [(0,0)],
        180: [(0,0)],
        270: [(0,0)],
    }

class TetrisEnvironment():

    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }

    def __init__(self, blocks: list[dict[int, list[tuple[int, int]]]] = TetrisBlocks.DEFAULTBLOCKS, width = 10, height = 20):
        self.tetrominos = blocks
        self.boardWidth = width
        self.boardHeight = height

    def reset(self):
        '''Resets the game, returning the current state'''
        self.board = [[TetrisEnvironment.MAP_EMPTY] * self.boardWidth] * self.boardHeight
        self.game_over = False
        self.bag = list(range(len(self.tetrominos)))
        random.shuffle(self.bag)
        self.next_piece = self.bag.pop()
        self._new_round()
        self.score = 0
        return self._get_board_props(self.board)


    def _get_rotated_piece(self):
        '''Returns the current piece, including rotation'''
        return self.tetrominos[self.current_piece][self.current_rotation]


    def _get_complete_board(self):
        '''Returns the complete board, including the current piece'''
        piece = self._get_rotated_piece()
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y][x] = TetrisEnvironment.MAP_PLAYER
        return board


    def get_game_score(self):
        '''Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        '''
        return self.score
    

    def _new_round(self):
        '''Starts a new round (new piece)'''
        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(self.tetrominos)))
            random.shuffle(self.bag)
        
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if self._check_collision(self._get_rotated_piece(), self.current_pos):
            self.game_over = True


    def _check_collision(self, piece, pos):
        '''Check if there is a collision between the current piece and the board'''
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= self.boardWidth \
                    or y < 0 or y >= self.boardHeight \
                    or self.board[y][x] == TetrisEnvironment.MAP_BLOCK:
                return True
        return False


    def _rotate(self, angle):
        '''Change the current rotation'''
        r = self.current_rotation + angle

        if r == 360:
            r = 0
        if r < 0:
            r += 360
        elif r > 360:
            r -= 360

        self.current_rotation = r


    def _add_piece_to_board(self, piece, pos):
        '''Place a piece in the board, returning the resulting board'''        
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = TetrisEnvironment.MAP_BLOCK
        return board


    def _clear_lines(self, board):
        '''Clears completed lines in a board'''
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == self.boardWidth]
        if lines_to_clear:
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(self.boardWidth)])
        return len(lines_to_clear), board


    def _column_heights(self, board):
        heights = [0] * self.boardWidth

        for i in range(self.boardWidth):
            j = 0
            while j < self.boardHeight and board[j][i] != TetrisEnvironment.MAP_BLOCK:
                j += 1
            heights[i] = self.boardHeight - j

        return heights


    def _height(self, board):
        heights = self._column_heights(board)
        return sum(heights), max(heights), min(heights)


    def _number_of_holes(self, board):
        '''Number of holes in the board (empty sqquare with at least one block above it)'''
        holes = 0
        covers = 0

        for col in zip(*board):
            i = 0
            while i < self.boardHeight and col[i] != TetrisEnvironment.MAP_BLOCK:
                i += 1
            holes += len([x for x in col[i+1:] if x == TetrisEnvironment.MAP_EMPTY])

            i = self.boardHeight - 1
            while i >= 0 and col[i] != TetrisEnvironment.MAP_EMPTY:
                i -= 1
            covers += len([x for x in col[:i+1] if x == TetrisEnvironment.MAP_BLOCK])

        return holes, covers


    def _bumpiness(self, board):
        heights = self._column_heights(board)
        bumps = [abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1)]
        return sum(bumps), max(bumps)


    def _exposure(self, board):
        totalExposure = 0
        blocks = 0
        for y in range(self.boardHeight):
            for x in range(self.boardWidth):
                if board[y][x] == TetrisEnvironment.MAP_BLOCK:
                    blocks += 1
                    totalExposure += board[y][min(self.boardWidth - 1, x + 1)] != TetrisEnvironment.MAP_BLOCK
                    totalExposure += board[y][max(0, x - 1)] != TetrisEnvironment.MAP_BLOCK
                    totalExposure += board[min(self.boardHeight - 1, y + 1)][x] != TetrisEnvironment.MAP_BLOCK
                    totalExposure += board[max(0, y - 1)][x] != TetrisEnvironment.MAP_BLOCK
        avgExposure = totalExposure / max(blocks, 1)
        return totalExposure, avgExposure


    def _get_board_props(self, board):
        '''Get properties of the board'''
        lines, board = self._clear_lines(board)
        holes, covers = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        total_exposure, avg_exposure = self._exposure(board)
        # return [lines, total_exposure, holes, max_height]
        return [lines, holes, total_bumpiness, sum_height]


    def get_next_states(self):
        '''Get all possible next states'''
        states = {}
        piece_id = self.current_piece
        
        if piece_id == 6: 
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = self.tetrominos[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, self.boardWidth - max_x):
                pos = [x, 0]

                # Drop piece
                while not self._check_collision(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(x, rotation)] = self._get_board_props(board)

        return states


    def get_state_size(self):
        '''Size of the state'''
        return 4


    def get_all_actions(self):
        rotations = [0, 90, 180, 270]
        return [(x, r) for r in rotations for x in range(-max([min(p[0] for p in t[r]) for t in self.tetrominos]), self.boardWidth - min([max(p[0] for p in t[r]) for t in self.tetrominos]))]


    def play(self, x, rotation, render=False, render_delay=None):
        '''Makes a play given a position and a rotation, returning the reward and if the game is over'''
        self.current_pos = [x, 0]
        self.current_rotation = rotation

        # Drop piece
        while not self._check_collision(self._get_rotated_piece(), self.current_pos):
            if render:
                self.render()
                if render_delay:
                    sleep(render_delay)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1

        # Update board and calculate score        
        self.board = self._add_piece_to_board(self._get_rotated_piece(), self.current_pos)
        # linesToSend = self._getGarbageLines(self._get_rotated_piece(), self.current_pos)
        lines_cleared, self.board = self._clear_lines(self.board)
        # score = 1 + (lines_cleared ** 2) * self.boardWidth
        score = lines_cleared
        self.score += score

        # Start new round
        self._new_round()
        if self.game_over:
            score = -1

        return score, self.game_over


    def _getGarbageLines(self, piece, pos):
        garbageLines = []
        for i in self.boardHeight:
            if sum(self.board[i]) == self.boardWidth:
                line = [TetrisEnvironment.MAP_BLOCK] * self.boardWidth
                for x, y in piece:
                    if y + pos[1] == i:
                        line[x + pos[0]] = self.MAP_EMPTY
                garbageLines += line
        return garbageLines


    def render(self):
        '''Renders the current board'''
        img = [TetrisEnvironment.COLORS[p] for row in self._get_complete_board() for p in row]
        img = np.array(img).reshape(self.boardHeight, self.boardWidth, 3).astype(np.uint8)
        img = img[..., ::-1] # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((self.boardWidth * 45, self.boardHeight * 45), Image.NEAREST)
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        cv2.waitKey(1)

