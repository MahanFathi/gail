import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# DISCRIMINATOR 
# ---------------------------------------------------------------------------- #
_C.DISC = CN()
_C.DISC.HIDD_SIZES = [32, 32]
_C.DISC.LR = 1e-3



def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
