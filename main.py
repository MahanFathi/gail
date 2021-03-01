import os
import pathlib
import pickle
import argparse
from datetime import datetime

import stable_baselines3 as sb3

from imitation.data import rollout
from imitation.util import util, logger

from gail.gail import GAIL
from config.defaults import get_cfg_defaults


def main():
    parser = argparse.ArgumentParser(description="Generative Adversarial Imitation Learning")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to yaml config file",
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()


    # Load pickled test demonstrations.
    with open(cfg.DATA.EXPERT_PATH, "rb") as f:
        # This is a list of `imitation.data.types.Trajectory`, where
        # every instance contains observations and actions for a single expert
        # demonstration.
        trajectories = pickle.load(f)

    transitions = rollout.flatten_trajectories(trajectories)

    # Create Environment
    venv = util.make_vec_env(cfg.ENV.GYM_ENV, n_envs=cfg.ENV.N_ENVS)

    # Logging
    config_file_name = os.path.basename(os.path.normpath(args.config_file)).split(".")[0]
    logger.configure(
        pathlib.Path("./logs") /
        config_file_name /
        str(datetime.now().time()),
        ["stdout", "log", "csv", "tensorboard"],
    )

    # Create Generator
    gen_algo = sb3.PPO(
        "MlpPolicy", venv, verbose=1, # venv is going to be updated later
        n_steps=cfg.GEN.N_STEPS,
        learning_rate=cfg.GEN.LR,
    )

    # Create GAIL learner
    gail_trainer = GAIL(
        cfg,
        venv,
        expert_data=transitions,
        gen_algo=gen_algo,
    )

    # Train
    gail_trainer.train(total_timesteps=cfg.GAIL.TOTAL_TIMESTEPS)


if __name__ == "__main__":
    main()
