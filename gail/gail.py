import dataclasses

import numpy as np
import torch as th
import torch.utils.data as th_data

import gym

from typing import Callable, Dict, Iterable, Mapping, Optional, Type, Union


from yacs.config import CfgNode
from stable_baselines3.common import on_policy_algorithm, preprocessing, vec_env

from imitation.data import buffer, types, wrappers
from imitation.util import reward_wrapper, util

from gail.discriminator import Discriminator


class GAIL(object):

    def __init__(
            self,
            cfg: CfgNode,
            venv: vec_env.VecEnv,
            expert_data: Union[Iterable[Mapping], types.Transitions],
            expert_batch_size: int,
            gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
            *,
            discrim_kwargs: Optional[Mapping] = None,
            **kwargs,
    ):
        self.cfg = cfg
        self.venv = venv
        self.expert_data = expert_data
        self.expert_batch_size = expert_batch_size
        self.gen_algo = gen_algo

        self._global_step = 0

        self._build()


    def _build(self):
        self._build_discriminator()
        self._wrap_env()
        self._build_buffer()
        self._build_expert_dataloader()
        self._build_disc_optimizer()


    def _build_discriminator(self):
        self.disc = Discriminator(self.cfg, self.venv.observation_space, self.venv.action_space)
        self.disc._logits_net.to('cuda')


    def _wrap_env(self, ):
        """To buffer data and replace reward function
        """
        self.venv = wrappers.BufferingWrapper(self.venv)
        # not normalizing env for now
        # self.venv_norm_obs = vec_env.VecNormalize(
        #     self.venv_buffering,
        #     norm_reward=False,
        #     norm_obs=normalize_obs,
        # )
        self.venv = reward_wrapper.RewardVecEnvWrapper(self.venv, self.disc.get_reward_np)
        # FIXME: can also normalize rewards
        # self.venv_train = vec_env.VecNormalize(
        #     self.venv_wrapped, norm_obs=False, norm_reward=normalize_reward
        # )
        self.gen_algo.set_env(self.venv)


    def _build_buffer(
            self,
    ):
        gen_replay_buffer_capacity = self.gen_batch_size
        self._gen_replay_buffer = buffer.ReplayBuffer(
            gen_replay_buffer_capacity, self.venv
        )


    def _build_expert_dataloader(self, ):
        # TODO: not quite familiar with data types
        self.expert_data_loader = th_data.DataLoader(
            self.expert_data,
            batch_size=self.expert_batch_size,
            collate_fn=types.transitions_collate_fn,
            shuffle=True,
            drop_last=True,
        )
        self._endless_expert_iterator = util.endless_iter(self.expert_data_loader)


    def _build_disc_optimizer(self, ):
        self.disc_optimizer = th.optim.Adam(self.disc._logits_net.parameters(), self.cfg.DISC.LR)


    def _torchify_array(self, ndarray: np.ndarray, **kwargs) -> th.Tensor:
        return th.as_tensor(ndarray, device='cuda', **kwargs)


    def _torchify_with_space(
        self, ndarray: np.ndarray, space: gym.Space, **kwargs
    ) -> th.Tensor:
        tensor = th.as_tensor(ndarray, device='cuda', **kwargs)
        preprocessed = preprocessing.preprocess_obs(tensor, space)
        return preprocessed


    @property
    def gen_batch_size(self, ) -> int:
        return self.gen_algo.n_steps * self.gen_algo.get_env().num_envs


    def train_gen(self, ):
        self.gen_algo.learn(
            total_timesteps=self.gen_batch_size, # upper bound on total transitions (i.e. gen dataset size)
            reset_num_timesteps=False,
        )
        self._global_step += 1

        gen_samples = self.venv.pop_transitions()
        self._gen_replay_buffer.store(gen_samples)


    def _make_disc_train_batch(self, ) -> Mapping:
        # create expert batch
        expert_samples = next(self._endless_expert_iterator)
        # create gen batch
        gen_samples = self._gen_replay_buffer.sample(self.expert_batch_size)
        gen_samples = types.dataclass_quick_asdict(gen_samples)

        # make sure they're dict
        expert_samples = dict(expert_samples)
        gen_samples = dict(gen_samples)

        # make them numpy
        for field in dataclasses.fields(types.Transitions):
            k = field.name
            if k == "infos":
                continue
            for d in [gen_samples, expert_samples]:
                if isinstance(d[k], th.Tensor):
                    d[k] = d[k].detach().numpy()

        n_gen = len(gen_samples["obs"])
        n_expert = len(expert_samples["obs"])

        # Concatenate rollouts, and label each row as expert or generator.
        obs = np.concatenate([expert_samples["obs"], gen_samples["obs"]])
        acts = np.concatenate([expert_samples["acts"], gen_samples["acts"]])
        next_obs = np.concatenate([expert_samples["next_obs"], gen_samples["next_obs"]])
        dones = np.concatenate([expert_samples["dones"], gen_samples["dones"]])
        labels_gen_is_one = np.concatenate(
            [np.zeros(n_expert, dtype=int), np.ones(n_gen, dtype=int)]
        )

        # TODO: Policy and reward network were trained on normalized observations.
        # obs = self.venv_norm_obs.normalize_obs(obs)
        # next_obs = self.venv_norm_obs.normalize_obs(next_obs)

        batch_dict = {
            "state": self._torchify_with_space(obs, self.disc.observation_space),
            "action": self._torchify_with_space(acts, self.disc.action_space),
            "next_state": self._torchify_with_space(next_obs, self.disc.observation_space),
            "done": self._torchify_array(dones),
            "labels_gen_is_one": self._torchify_array(labels_gen_is_one),
        }

        return batch_dict


    def train_disc(self, ):

        batch = self._make_disc_train_batch()
        disc_loss = self.disc.get_loss(
            batch['state'],
            batch['action'],
            batch['labels_gen_is_one'],
        )
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()


    def train(self, total_timesteps):

        n_rounds = total_timesteps // self.gen_batch_size

        for _ in range(n_rounds):
            self.train_gen()
            for _ in range(self.cfg.DISC.UPDATES_PER_ROUND):
                self.train_disc()
