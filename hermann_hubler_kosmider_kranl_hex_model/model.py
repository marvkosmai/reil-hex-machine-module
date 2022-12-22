from .hex.MCTS import MCTS
from .hex.HexGame import HexGame
from .hex.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from .utils import *


class Model:
    def __init__(self):
        self.g = HexGame(7)
        self.n1 = NNet(self.g)
        self.n1.load_checkpoint('./hermann_hubler_kosmider_kranl_hex_model/7x7/', 'best.pth.tar')
        self.args1 = dotdict({'numMCTSSims': 25, 'cpuct': 1.0})
        self.mcts1 = MCTS(self.g, self.n1, self.args1)
        self.n1p = lambda x, player: np.argmax(self.mcts1.getActionProb(x, player, temp=0))

    def machine(self, board, action_list):
        board = np.array(board)
        board[board == 2] = -1
        player = -1 if len(action_list) % 2 == 0 else 1
        action = self.n1p(board, player)

        # flip and rotate action
        if player == -1:
            space = np.zeros(7 * 7)
            space[action] = 1
            space = np.reshape(space, (7, 7))
            space = np.fliplr(space)
            space = np.rot90(space)
            space = space.flatten()
            for i in range(len(space)):
                if space[i] == 1:
                    action = i
                    break

        return int(action / 7), action % 7
