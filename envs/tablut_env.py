import numpy as np
import gym
from gym import spaces
from tablutpy.tablut.rules.ashton import Board
from enum import Enum
from tablutpy.tablut.board import DrawException, LoseException, WinException


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

    @property
    def PACKED_OBSERVATION(self):
        return {
            "te": 0,
            "TW": 1,
            "TB": 2,
            "TK": 3,
            "ce": 4,
            "CB": 5,
            "ee": 6,
            "EK": 7,
            "Se": 8,
            "SK": 9
        }

    def __init__(self):
        self.board = Board()
        self.turn = self.FIRST_PLAYER
        self.action_space = spaces.MultiDiscrete([9, 9, 9, 9])
        self.observation_space = spaces.MultiDiscrete([10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                                       10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])

    def reset(self):
        """
        This should return the observation of the state.
        """
        return self.get_observation_space()

    def get_observation_space(self):
        packed_board = self.board.pack(self.board.board)
        observation = []
        for row in packed_board:
            for col in row:
                observation.append(self.PACKED_OBSERVATION[col])
        return np.array(observation)

    def step(self, action):
        """
        I imagine the action here to be a tuple composed of ((startx, starty), (endx, endy))
        So we decompose that.
        Then, we'd like to define an "intermediate" reward for captures and a BIG ass reward for winning. I've kind of done that, check ashton.py and board.py
        """
        done = False
        try:
            reward, done = self.board.step(
                (action[0], action[1]), (action[2], action[3]))
        except DrawException:
            reward = 0
            done = True
        except ValueError:  # If the step was wrong, negative reward
            reward = -1
            done = False
        except Exception as e:
            print(e)
            reward = 0
            done = False
        info = {}
        return self.get_observation_space(), reward, done, info

    def render(self, mode='console'):
        print(self.board.pack(self.board.board))
