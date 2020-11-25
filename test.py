import os
import sys
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import A2C
from envs import AshtonWhitePlayerEnv

from tablut.game import Game
from tablut.board import WinException, DrawException, LoseException
from tablut.rules.ashton import Board, Player
from tablut.player import RandomPlayer

if __name__ == "__main__":
    model = A2C.load(os.path.join(os.getcwd(), "ashton_white_player.zip"))

    game = Game(Board())
    opponent = RandomPlayer(game, Player.BLACK)

    # Testing the agent
    # Try 5000 moves and choose the first legal obtained
    for i in range(1000):
        action, _state = model.predict(AshtonWhitePlayerEnv.board_to_np(
            game.board.board), deterministic=True)
        
        start_row, start_col, end_row, end_col = action
        start = (start_row, start_col)
        end = (end_row, end_col)
        if game.board.is_legal(Player.WHITE, start, end):
            try:
                game.white_move(start, end)
            except WinException:
                print("WIN!")
                sys.exit(0)
            except LoseException:
                print("LOSE!")
                sys.exit(0)
            except DrawException:
                print("DRAW!")
                sys.exit(0)