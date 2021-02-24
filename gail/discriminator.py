import numpy as np

import torch as th
import torch.nn as nn
import torch.functional as F

import gym

from yacs.config import CfgNode

from stable_baselines3.common import preprocessing, vec_env

from imitation.rewards.discrim_nets import ActObsMLP
from util import networks


class Discriminator(object):
    def __init__(
            self,
            cfg: CfgNode,
            obs_space: gym.Space,
            act_space: gym.Space,
    ):
        self.cfg = cfg
        self.observation_space = obs_space
        self.action_space = act_space


    def _build(self, ):
        self._build_net

    def _build_net(self, ):
       hidden_sizes = cfg.DISC.HIDD_SIZES
       self._logits_net = ActObsMLP(self.action_space, self.observation_space, hid_sizes=hidden_sizes)


    def get_logits(
            self,
            observations: th.Tensor,
            actions: th.Tensor,
    ):
        return self._logits_net(observations, actions)


    def get_loss(
            self,
            observations: th.Tensor,
            actions: th.Tensor,
            labels_gen_is_one: th.Tensor,
    ):
        logits_gen_is_high = self.get_logits(observations, actions)
        return F.binary_cross_entropy_with_logits(
            logits_gen_is_high, labels_gen_is_one.float()
        )



    def get_reward(
            self,
            state: th.Tensor,
            action: th.Tensor,
            next_state: th.Tensor,
            done: th.Tensor,
    ) -> th.Tensor:
        # note: gen logits are high
        logits = self.get_logits(state, action)
        rew = -F.logsigmoid(logits)
        return rew


    def get_reward_np(
            self,
            state: np.ndarray,
            action: np.ndarray,
            next_state: np.ndarray,
            done: np.ndarray,
            device='cuda',
            scale: bool = False,
    ) -> np.ndarray:
        state_th = th.as_tensor(state, device=device)
        action_th = th.as_tensor(action, device=device)
        next_state_th = th.as_tensor(next_state, device=device)

        # FIXME: no preprocessing for now
        # state_th = preprocessing.preprocess_obs(state_th, observation_space, scale)
        # action_th = preprocessing.preprocess_obs(action_th, action_space, scale)
        # next_state_th = preprocessing.preprocess_obs(
        #     next_state_th, observation_space, scale
        # )

        done_th = th.as_tensor(done, device=device)
        done_th = done_th.to(th.float32)

        with th.no_grad():
            rew_th = self.get_reward(state_th, action_th)

        # rewards are flattened to match the trajectory size
        rew = rew_th.detach().cpu().numpy().flatten()

        return rew
