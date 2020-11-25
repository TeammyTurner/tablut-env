import os
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3 import A2C
from envs import AshtonWhitePlayerEnv

if __name__ == "__main__":        
    env = AshtonWhitePlayerEnv()
    env = make_vec_env(lambda: env, n_envs=1)
    
    model = A2C('MlpPolicy', env, verbose=1)
    
    model.learn(total_timesteps=10000)
    # save in current working directory
    model.save(os.getcwd() / "ashton_white_player.zip")