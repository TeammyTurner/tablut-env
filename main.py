from envs.tablut_env import TablutEnv

from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env


env = TablutEnv()
env = make_vec_env(lambda: env, n_envs=1)
model = ACKTR('MlpPolicy', env, verbose=1).learn(50000)
# Test the trained agent
obs = env.reset()
print(obs)
n_steps = 100
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    env.render(mode='console')
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break
