import numpy as np
import sys
sys.path.append('../..')
from .Game import Game
from .HexLogic import Board


class HexGame(Game):

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.board)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n * self.n

    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.board = np.copy(board)
        move = (int(action / self.n), action % self.n)
        b.playAction(move, player)
        return (b.board, -player)

    def getValidMoves(self, board, player):
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.board = np.copy(board)
        legalMoves = b.getActionSpace()
        for x, y in legalMoves:
            valids[self.n*x+y] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        b = Board(self.n)
        b.board = np.copy(board)
        if b.whiteWin():
            return 1 * player

        if b.blackWin():
            return -1 * player

        return 0

    def getCanonicalForm(self, board, player):
        if player == 1:
            return board
        else:
            return np.fliplr(np.rot90(-1 * board, axes=(1, 0)))

    def getOriginalForm(self, board, player):
        if player == 1:
            return board
        else:
            return np.rot90(np.fliplr(-1 * board), axes=(0, 1))

    def getSymmetries(self, board, pi):
        assert (len(pi) == self.n ** 2)
        pi_board = np.reshape(pi, (self.n, self.n))
        l = []

        for i in [0, 2]:
            newB = np.rot90(board, i)
            newPi = np.rot90(pi_board, i)
            l += [(newB, list(newPi.ravel()))]
        return l

    def stringRepresentation(self, board):
        return board.tostring()

    @staticmethod
    def display(board):
        n = len(board)
        b = Board(n)
        b.board = np.copy(board)
        b.printBoard()
