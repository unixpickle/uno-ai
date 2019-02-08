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
        while game.winner() is None:
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
      log_probs: a (seq_len, batch) Tensor storing the
        initial log probabilities of the actions.
      masks: a (seq_len, batch, ACTION_VECTOR_SIZE) Tensor
        of values where a 1 indicates that an action was
        allowed, and a 0 indicates otherwise.
      advs: a (seq_len, batch) Tensor of advantages.
      targets: a (seq_len, batch) Tensor of target values.
      seq_mask: a (seq_len, batch) Tensor of values where
        a 1 indicates that the element is valid.
    """

    def __init__(self, rollouts, device, gamma=0.99, lam=0.95):
        seq_len = max(r.num_steps for r in rollouts)
        batch = len(rollouts)
        observations = np.zeros([seq_len, batch, OBS_VECTOR_SIZE], dtype=np.float32)
        actions = np.zeros([seq_len, batch, ACTION_VECTOR_SIZE], dtype=np.float32)
        log_probs = np.zeros([seq_len, batch], dtype=np.float32)
        masks = np.zeros([seq_len, batch, ACTION_VECTOR_SIZE], dtype=np.float32)
        advs = np.zeros([seq_len, batch], dtype=np.float32)
        targets = np.zeros([seq_len, batch], dtype=np.float32)
        seq_mask = np.zeros([seq_len, batch], dtype=np.float32)
        for i, r in enumerate(rollouts):
            observations[:r.num_steps, i, :] = r.observations
            actions[:r.num_steps, i, :] = [_one_hot_action(o['action']) for o in r.outputs]
            actions[r.num_steps:, i, 0] = 1
            log_probs[:r.num_steps, i] = [o['log_prob'] for o in r.outputs]
            masks[:r.num_steps, i, :] = [_option_mask(o['options']) for o in r.outputs]
            masks[r.num_steps:, i, 0] = 1
            our_advs = r.advantages(gamma=gamma, lam=lam)
            advs[:r.num_steps, i] = our_advs
            targets[:r.num_steps, i] = [adv + o['value'] for adv, o in zip(our_advs, r.outputs)]
            seq_mask[:r.num_steps, i] = 1

        def proc_list(l):
            return torch.from_numpy(l).to(device)

        self.observations = proc_list(observations)
        self.actions = proc_list(actions)
        self.log_probs = proc_list(log_probs)
        self.masks = proc_list(masks)
        self.advs = proc_list(advs)
        self.targets = proc_list(targets)
        self.seq_mask = proc_list(seq_mask)

    def masked_mean(self, seqs):
        """
        Compute a mean using the sequence mask.

        Args:
            seqs: a (seq_len, batch) Tensor.

        Returns:
            A masked mean.
        """
        return torch.sum(seqs * self.seq_mask) / torch.sum(self.seq_mask)


def _one_hot_action(action):
    res = [0.0] * ACTION_VECTOR_SIZE
    res[action.index()] = 1.0
    return res


def _option_mask(options):
    res = [0.0] * ACTION_VECTOR_SIZE
    for a in options:
        res[a.index()] = 1.0
    return res
