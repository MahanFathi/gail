import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# Model Configs
# ---------------------------------------------------------------------------- #
_C.POLICY = CN()
_C.POLICY.ALGORITHM = 'VanillaPG'
_C.POLICY.NN_INTERMEDIATE_LAYER_SIZES = [32, 32, 8]

# ---------------------------------------------------------------------------- #
# Vanilla Policy Gradients Specific Config
# ---------------------------------------------------------------------------- #
_C.POLICY.VPG = CN()
_C.POLICY.VPG.LATENT_NN_INTERMEDIATE_LAYER_SIZES = [32, 32, 8]
_C.POLICY.VPG.LATENT_NN_ACTIONS_OUT_SIZES = [8]
_C.POLICY.VPG.LATENT_NN_VALUE_OUT_SIZES = [8]

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
