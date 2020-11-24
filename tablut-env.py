import numpy as np
import gym
from gym import spaces
from tablutpy.tablut.rules.ashton import Board


class Turn(Enum):
    WHITE = "W"
    BLACK = "B"


class TablutEnv(gym.Env):
    """
    We want to comment on the observation space since its kinda esoteric.
    Let's consider the 9x9 board, we know that in each of the positions we can either find a piece of certain type or nothing.
    We can therefore map the observation space as a 3D discrete vector, composed of [X, Y, what's in there].
    Therefore, we map the piece types to:
            0: (Tile, board.EmptyTile),
            1: (Tile, WhiteSoldier),
            2: (Tile, BlackSoldier),
            3: (Tile, King),
            4: (Camp, board.EmptyTile),
            5: (Camp, BlackSoldier),
            6: (Escape, board.EmptyTile),
            7: (Escape, King),
            8: (Castle, board.EmptyTile),
            9: (Castle, King)
    """

    metadata = {'render.modes': ['console']}

    FIRST_PLAYER = Turn.WHITE

    def __init__(self):
        self.board = Board()
        self.turn = FIRST_PLAYER
        self.action_space = spaces.MultiDiscrete([9, 9, 9, 9])
        self.observation_space = spaces.MultiDiscrete([9, 9, ])

    def reset(self):
        """
        This should return the observation of the state.
        The thing is: how the fuck do we represent this giant ass space?
        """
        pass

    def step(self, action):
        """
        I imagine the action here to be a tuple composed of ((startx, starty), (endx, endy))
        So we decompose that.
        Then, we'd like to define an "intermediate" reward for captures and a BIG ass reward for winning. I've kind of done that, check ashton.py and board.py
        """
        done = False
        obs, reward, done = self.board.step(
            (action[0], action[1]), (action[2], action[3]))
        info = None
        return obs, reward, done, info

    def render(self, mode='console'):
        print(self.board.pack())
