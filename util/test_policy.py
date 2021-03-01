import os
import gym
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import VecNormalize

from imitation.util import util, logger

def test_policy(policy: sb3.PPO, iteration: int, model_path=None, env_stats_path=None):

    model_path = model_path or os.path.join("./models", "{}.zip".format(iteration))
    env_stats_path = env_stats_path or os.path.join("./envs", "{}.pkl".format(iteration))

    env = policy.get_env()
    env = VecNormalize.load(env_stats_path, env)
    env.training = False
    env.norm_reward = False

    policy.load(model_path)

    obs = env.reset()
    for _ in range(1000):
        action, _states = policy.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
