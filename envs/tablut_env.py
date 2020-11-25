import copy
from enum import Enum
import numpy as np
import gym
from tablut.board import EmptyTile, WinException, LoseException, DrawException
from tablut.rules.ashton import WhiteSoldier, BlackSoldier, King, Tile, Camp, Escape, Castle, Board
from tablut.game import Game


class Turn(Enum):
    WHITE = "W"
    BLACK = "B"


class TablutWhitePlayerEnv(gym.Env):
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
    BOARD_MAPPING = {
        "0": (Tile, EmptyTile),
        "1": (Tile, WhiteSoldier),
        "2": (Tile, BlackSoldier),
        "3": (Tile, King),
        "4": (Camp, EmptyTile),
        "5": (Camp, BlackSoldier),
        "6": (Escape, EmptyTile),
        "7": (Escape, King),
        "8": (Castle, EmptyTile),
        "9": (Castle, King)
    }

    metadata = {'render.modes': ['console']}

    FIRST_PLAYER = Turn.WHITE

    def __init__(self):
        self.game = Game(Board())
        
        #self.action_space = spaces.MultiDiscrete([9, 9, 9, 9])
        #self.observation_space = spaces.MultiDiscrete([9, 9, ])
        
        # Blindly based on https://github.com/towzeur/gym-abalone/blob/master/gym_abalone/envs/abalone_env.py
        # should be a 9x9 matrix of integers in the domain [0, 9]
        # Some algorithms supports only Box, some only discrete... which one should we use?
        # (see https://stable-baselines.readthedocs.io/en/master/guide/algos.html)
        
        # action_space is modeled through a Box which is indeed a 4D vector, the first couple represents 
        # starting row and column, the second couple ending row and column of the move    
        self.action_space = gym.spaces.Box(0, 8, shape=(4,), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(np.int8(0), np.int8(9), shape=(9, 9), dtype=np.int8)

    @property
    def turn(self):
        """
        The current game turn
        """
        return self.game.turn

    @property
    def board(self):
        """
        The current game board configuration
        """
        return self.game.board

    def reset(self):
        """
        This should return the observation of the state.
        The thing is: how the fuck do we represent this giant ass space?
        """
        # TODO: reset the game board etc
        self.game = Game(Board())
        return self.observation

    def step(self, action):
        """
        Perform a step in the environment, to start trying out im just gonna give +1 when a move is legal
        and -1 when a move is not legal.
        TODO: Add an opponent, at least making random moves.
        """
        info = {}
        done = False
        reward = 0

        start_x, start_y, end_x, end_y = action

        try:
            self.game.white_move((start_x, start_y), (end_x, end_y))
            observation = self.observation
            # legal move performed
            reward += 1
        except ValueError:
            # illegal move
            reward = -1
        except (WinException, LoseException, DrawException):
            # game ended
            done = True

        return self.observation, reward, done, info

    @property
    def observation(self):
        """
        Encode game board in a numpy array of shape (8, 8) using the mapping defined in BOARD_MAPPING
        """
        # Use the dict in inverse order: tile content -> code
        codes = list(self.BOARD_MAPPING.keys())
        keys = list(self.BOARD_MAPPING.values())

        grid = np.zeros((9, 9), dtype=np.int8)
        for row_i, _ in enumerate(self.board.board):
            for col_i, _ in enumerate(self.board.board[row_i]):
                tile = self.board.board[row_i][col_i]
                # search for a key whose content matches the tile's content
                matching_key = [
                    isinstance(tile, k[0]) and isinstance(tile.piece, k[1])
                    for k in keys
                ]
                code_idx = matching_key.index(True)
                # set the corresponding value on the grid
                grid[row_i, col_i] = np.int8(codes[code_idx])

        return grid

    def render(self, mode='console'):
        print(self.board.pack())

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = TablutWhitePlayerEnv()
    check_env(env)