import os
import logging
import dataclasses
from typing import Callable, Dict, Iterable, Mapping, Optional, Type, Union

import gym
import numpy as np
import torch as th
import torch.utils.data as th_data
import torch.utils.tensorboard as thboard
from yacs.config import CfgNode

from stable_baselines3.common import on_policy_algorithm, preprocessing, vec_env

from imitation.data import buffer, types, wrappers
from imitation.util import reward_wrapper, util, logger
from imitation.rewards import common as rew_common

from gail.discriminator import Discriminator


class GAIL(object):

    def __init__(
            self,
            cfg: CfgNode,
            venv: vec_env.VecEnv,
            expert_data: Union[Iterable[Mapping], types.Transitions],
            gen_algo: on_policy_algorithm.OnPolicyAlgorithm,
            log_dir: Optional[types.AnyPath] = None,
    ):

        self.cfg = cfg
        self.venv = venv
        self.expert_data = expert_data
        self.gen_algo = gen_algo

        self.expert_batch_size = cfg.DATA.EXPERT_BATCH_SIZE
        self.log_dir = log_dir

        self._global_step = 0
        self._disc_step = 0

        self._build()


    def _build(self, ):
        # self._build_tensorboard()
        self._build_discriminator()
        self._wrap_env()
        self._build_buffer()
        self._build_expert_dataloader()
        self._build_disc_optimizer()


    def _build_tensorboard(self):
        logging.info("building summary directory at " + "./output")
        summary_dir = os.path.join("./output", "summary")
        os.makedirs(summary_dir, exist_ok=True)
        self._summary_writer = thboard.SummaryWriter(summary_dir)


    def _build_discriminator(self, ):
        self.disc = Discriminator(self.cfg, self.venv.observation_space, self.venv.action_space)
        self.disc._logits_net.to('cuda')


    def _wrap_env(self, ):
        """To buffer data and replace reward function
        """

        # buffering wrapper is the first layer so we can store raw obs&rew
        self.venv = wrappers.BufferingWrapper(self.venv)

        # only normalize obs, disc reward wrapper should be next
        # NOTE: `venv_norm_obs`: given a specific name, since this wrapper is used twice
        self.venv_norm_obs = vec_env.VecNormalize(
            self.venv,
            norm_reward=False,
            norm_obs=self.cfg.GAIL.NORM_OBS,
        )

        # reward returned by disc are now based on normalized obs
        self.venv = reward_wrapper.RewardVecEnvWrapper(self.venv_norm_obs, self.disc.get_reward_np)

        # these rewards are now normalized for gen use
        self.venv = vec_env.VecNormalize(
            self.venv,
            norm_reward=self.cfg.GAIL.NORM_REW,
            norm_obs=False,
        )

        self.gen_algo.set_env(self.venv)


    def _build_buffer(self, ):
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


    def dump_gen_and_env(self, ) -> None:
        # checkpoint for generator (i.e. policy)
        if self._global_step % self.cfg.GEN.MODEL_DUMP_PERIOD is not 0:
            return
        os.makedirs(self.log_dir / "models", exist_ok=True)
        filepath = self.log_dir / "models" / "{}".format(self._global_step)
        self.gen_algo.save(filepath)

        # checkpoints for envs are necessary when observations are normalized
        if not self.cfg.GAIL.NORM_OBS:
            return
        os.makedirs(self.log_dir / "envs", exist_ok=True)
        filepath = self.log_dir / "envs" / "{}.pkl".format(self._global_step)
        self.venv_norm_obs.save(filepath)


    def train_gen(self, ):

        with logger.accumulate_means("gen"):
            self.gen_algo.learn(
                total_timesteps=self.gen_batch_size, # upper bound on total transitions (i.e. gen dataset size)
                reset_num_timesteps=False,
            )

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

        # NOTE: expert's and gen's (unnormalized gen's) data should be normalized now
        obs = self.venv_norm_obs.normalize_obs(obs)
        next_obs = self.venv_norm_obs.normalize_obs(next_obs)

        batch_dict = {
            "state": self._torchify_with_space(obs, self.disc.observation_space),
            "action": self._torchify_with_space(acts, self.disc.action_space),
            "next_state": self._torchify_with_space(next_obs, self.disc.observation_space),
            "done": self._torchify_array(dones),
            "labels_gen_is_one": self._torchify_array(labels_gen_is_one),
        }

        return batch_dict


    def train_disc(self, ):

        with logger.accumulate_means("disc"):

            should_write_summaries = self._global_step % 20 == 0

            batch = self._make_disc_train_batch()
            logits_gen_is_high = self.disc.get_logits(batch["state"], batch["action"])
            disc_loss = self.disc.get_loss(
                logits_gen_is_high,
                batch['labels_gen_is_one'],
            )
            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()
            self._disc_step += 1

            # compute/write stats and TensorBoard data
            with th.no_grad():
                train_stats = rew_common.compute_train_stats(
                    logits_gen_is_high, batch["labels_gen_is_one"], disc_loss
                )
            logger.record("global_step", self._global_step)
            for k, v in train_stats.items():
                logger.record(k, v)
            logger.dump(self._disc_step)
            # if should_write_summaries:
                # self._summary_writer.add_histogram("disc_logits", logits_gen_is_high.detach())


    def train(self, total_timesteps):

        n_rounds = total_timesteps // self.gen_batch_size

        for _ in range(n_rounds):
            self._global_step += 1
            self.train_gen()
            for _ in range(self.cfg.DISC.UPDATES_PER_ROUND):
                self.train_disc()
            logger.dump(self._global_step)
            self.dump_gen_and_env()
