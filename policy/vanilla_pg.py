import gym
from yacs.config import CfgNode

from util.nn import FeedForward
from util.distributions import make_proba_distribution


class VanillaPG(object):
    def __init__(self, cfg: CfgNode, env: gym.Env):
        self.cfg = cfg
        self.env = env

        # TODO: fix to support all kinds of obs/act spaces
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]


        self.log_std_init = 0 # TODO: should come from cfg


    def _build(self, ):
        self._build_latent_net()
        self,_build_policy_nn()


    def _build_latent_net(self, ):

        self._latent_layer_sizes = cfg.POLICY.VPG.LATENT_NN_INTERMEDIATE_LAYER_SIZES
        self._latent_net_base = FeedForward(self.obs_dim, self._latent_layer_sizes)

        # FIXME: actions head is used as `self._latent_net_base` output
        # there's actually a linear part for action net in distributions
        # self._latent_action_out_size = cfg.POLICY.VPG.LATENT_NN_VALUE_OUT_SIZES
        # self._latent_net_actions = FeedForward(self._latent_layer_sizes[-1], [], self._latent_action_out_size)

        self.value_net = FeedForward(self._latent_layer_sizes[-1],
                                     cfg.POLICY.VPG.LATENT_NN_VALUE_OUT_SIZES,
                                     1)


    def _build_policy_nn(self, ):
        # TODO: onlu assuming `DiagGaussianDistribution` distribution
        self.action_dist = make_proba_distribution(self.env.action_space)

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            latent_dim=self._latent_layer_sizes,
            log_std_init=self.log_std_init
        )


    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, ) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(mean_actions, self.log_std)


    def _get_latent_pi(self, observations: torch.Tensor) -> torch.Tensor:
        return self._latent_net_base(observations) # TODO: for now latent_pi is no different than 'latent_all'


    def _get_latent_v(self, observations: torch.Tensor) -> torch.Tensor:
        return self._latent_net_base(observations) # TODO: for now latent_v is no different than 'latent_all'


    def get_actions(
            self, obs: torch.Tensor,
            deterministic_mode: bool, # should be off when training
    ):
        latent_pi = self._get_latent_pi(obs)
        distribution = self._get_action_dist_from_latent(latent_pi)
        return distribution.get_actions(deterministic_mode)


    def get_values(self, obs: torch.Tensor):
        latent_v = self._get_latent_v(obs)
        return self.value_net(latent_v)


    def train(self, data):
        pass
