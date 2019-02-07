"""
Gather trajectories from agents on a game.
"""

import numpy as np
import torch

from .actions import ACTION_VECTOR_SIZE
from .game import OBS_VECTOR_SIZE


class Rollout:
    """
    A single episode from the perspective of a single
    agent.

    Stores the observation for every timestamp, the output
    of the agent at every timestamp, and the reward of the
    final episode from the agent's perspective.
    """

    def __init__(self, observations, outputs, reward):
        self.observations = observations
        self.outputs = outputs
        self.reward = reward

    @property
    def num_steps(self):
        """
        Get the number of timesteps.
        """
        return len(self.observations)

    def advantages(self, gamma=0.99, lam=0.95):
        """
        Compute the advantages using Generalized
        Advantage Estimation.
        """
        res = []
        adv = 0
        for i in range(self.num_steps)[::-1]:
            if i == self.num_steps - 1:
                delta = self.reward - self.outputs[-1]['value']
            else:
                delta = gamma * self.outputs[i + 1]['value'] - self.outputs[i]['value']
            adv *= lam * gamma
            adv += delta
            res.append(adv)
        return res[::-1]

    @classmethod
    def empty(cls):
        return cls([], [], 0.0)

    @classmethod
    def rollout(cls, game, agents):
        rollouts = [cls.empty() for _ in agents]
        states = [None] * len(agents)
        while not game.winner():
            for i, agent in enumerate(agents):
                res = agent.step(game, i, states[i])
                states[i] = res['state']
                r = rollouts[i]
                r.observations.append(game.obs(i))
                r.outputs.append(res)
            game.act(rollouts[game.turn()].outputs[-1]['action'])
        for i, r in enumerate(rollouts):
            if i == game.winner():
                r.reward = 1.0
        return rollouts


class RolloutBatch:
    """
    A packed batch of rollouts.

    Specifically, stores the following:
      observations: a (seq_len, batch, OBS_VECTOR_SIZE)
        Tensor of observations.
      actions: a (seq_len, batch, ACTION_VECTOR_SIZE)
        Tensor of one-hot vectors indicating which action
        was taken at every timestep.
      log_probs: the initial log probabilities of the
        actions.
      masks: a (seq_len, batch, ACTION_VECTOR_SIZE) Tensor
        of values where a 1 indicates that an action was
        allowed, and a 0 indicates otherwise.
      advs: a (seq_len, batch) Tensor of advantages.
      targets: a (seq_len, batch) Tensor of target values.
    """

    def __init__(self, rollouts, device, gamma=0.99, lam=0.95):
        seq_len = max(r.num_steps for r in rollouts)
        observations = []
        actions = []
        log_probs = []
        masks = []
        advs = []
        targets = []
        for r in rollouts:
            obs_seq = r.observations.copy()
            act_seq = [_one_hot_action(o['action']) for o in r.outputs]
            log_seq = [o['log_prob'] for o in r.outputs]
            mask_seq = [_option_mask(o['options']) for o in r.outputs]
            adv_seq = r.advantages(gamma=gamma, lam=lam)
            targ_seq = [adv + o['value'] for adv, o in zip(adv_seq, r.outputs)]
            for _ in range(seq_len - r.num_steps):
                obs_seq.append([0] * OBS_VECTOR_SIZE)
                act_seq.append(_one_hot_action(0))
                log_seq.append(0)
                mask_seq.append(_one_hot_action(0))
                adv_seq.append(0)
                targ_seq.append(0)
            observations.append(obs_seq)
            actions.append(act_seq)
            log_probs.append(log_seq)
            masks.append(mask_seq)
            advs.append(adv_seq)
            targets.append(targ_seq)

        def proc_list(l):
            axes = [1, 0]
            if len(np.array(l).shape) == 3:
                axes.append(2)
            return torch.from_numpy(
                np.transpose(np.array(l, dtype=np.float32), axes=axes)).to(device)
        self.observations = proc_list(observations)
        self.actions = proc_list(actions)
        self.log_probs = proc_list(log_probs)
        self.masks = proc_list(masks)
        self.advs = proc_list(advs)
        self.targets = proc_list(targets)


def _one_hot_action(action):
    res = [0.0] * ACTION_VECTOR_SIZE
    res[action.index()] = 1.0
    return res


def _option_mask(options):
    res = [0.0] * ACTION_VECTOR_SIZE
    for a in options:
        res[a.index()] = 1.0
    return res
