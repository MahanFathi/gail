import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()


# ---------------------------------------------------------------------------- #
# ENVIRONMENT
# ---------------------------------------------------------------------------- #
_C.ENV = CN()
_C.ENV.GYM_ENV = "CartPole-v1"
_C.ENV.N_ENVS = 2

# ---------------------------------------------------------------------------- #
# DISCRIMINATOR 
# ---------------------------------------------------------------------------- #
_C.DISC = CN()
_C.DISC.HIDD_SIZES = [32, 32]
_C.DISC.LR = 1e-3
_C.DISC.UPDATES_PER_ROUND = 12

# ---------------------------------------------------------------------------- #
# GENERATOR
# ---------------------------------------------------------------------------- #
_C.GEN = CN()
_C.GEN.N_STEPS = 1024 # number of steps to run for each environment per update
_C.GEN.LR = 3e-4
_C.GEN.MODEL_DUMP_PERIOD = 10

# ---------------------------------------------------------------------------- #
# DATA
# ---------------------------------------------------------------------------- #
_C.DATA = CN()
_C.DATA.EXPERT_PATH = "./data/expert_models/cartpole_0/rollouts/final.pkl"
_C.DATA.EXPERT_BATCH_SIZE = 32

# ---------------------------------------------------------------------------- #
# GAIL
# ---------------------------------------------------------------------------- #
_C.GAIL = CN()
_C.GAIL.TOTAL_TIMESTEPS = 10 * 2048
_C.GAIL.NORM_OBS = True
_C.GAIL.NORM_REW = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
