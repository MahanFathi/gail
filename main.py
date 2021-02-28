import pathlib
import pickle
import tempfile

from datetime import datetime

import stable_baselines3 as sb3

from gail.gail import GAIL

from imitation.data import rollout
from imitation.util import util, logger

from config.defaults import get_cfg_defaults

cfg = get_cfg_defaults()

# Load pickled test demonstrations.
with open("./data/expert_models/cartpole_0/rollouts/final.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("CartPole-v1", n_envs=2)

# logging
logger.configure(pathlib.Path("./logs") / str(datetime.now().time())), ["stdout", "log", "csv", "tensorboard"])

gail_trainer = GAIL(
    cfg,
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
)
gail_trainer.train(total_timesteps=100 * 2048)
