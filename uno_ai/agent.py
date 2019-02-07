"""
Reinforcement Learning agents.
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn

from .actions import ACTION_VECTOR_SIZE, NopAction
from .game import OBS_VECTOR_SIZE


class Agent(nn.Module):
    """
    A stochastic policy plus a value function.
    """

    def __init__(self):
        super().__init__()
        self.input_proc = nn.Sequential(
            nn.Linear(OBS_VECTOR_SIZE, 256),
            nn.Tanh,
            nn.Linear(256, 256),
            nn.Tanh,
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2)
        self.policy = nn.Linear(256, ACTION_VECTOR_SIZE)
        self.value = nn.Linear(256, 1)

    def device(self):
        return self.parameters()[0].device

    def forward(self, inputs, states=None):
        """
        Apply the agent to a batch of sequences.

        Args:
            inputs: a (seq_len, batch, OBS_VECTOR_SIZE)
              Tensor of observations.
            states: a tuple (h_0, c_0) of states.

        Returns:
            A tuple (logits, states):
              logits: A (seq_len, batch, ACTION_VECTOR_SIZE)
                Tensor of logits.
              values: A (seq_len, batch) Tensor of values.
              states: a new (h_0, c_0) tuple.
        """
        seq_len, batch = inputs.shape[0], inputs.shape[1]
        flat_in = inputs.view(-1, OBS_VECTOR_SIZE)
        features = self.input_proc(flat_in)
        features = features.view(seq_len, batch, -1)
        if states is None:
            outputs, h_n, c_n = self.rnn(features)
        else:
            outputs, h_n, c_n = self.rnn(features, states)
        flat_out = outputs.view(-1, outputs.shape[-1])
        flat_logits = self.policy(flat_out)
        flat_values = self.value(flat_out)
        logits = flat_logits.view(seq_len, batch, ACTION_VECTOR_SIZE)
        values = flat_values.view(seq_len, batch)
        return logits, values, (h_n, c_n)

    def step(self, game, player, state):
        """
        Pick an action in the game.

        Args:
            game: the Game we are playing.
            player: the index of this agent.
            state: the previous RNN state (or None).

        Returns:
            A dict containing the following keys:
              options: the options chosen from.
              action: the sampled action.
              log_prob: the log probability of the action.
              value: the value function output.
              state: the new RNN state.
        """
        obs = torch.from_numpy(np.array(game.obs(player))).to(self.device())
        obs = obs.view(1, 1, -1)
        options = [NopAction()]
        if player == game.turn():
            options = game.options()
        vec, values, new_state = self(obs, states=state)
        np_vec = vec.view(-1).detach().cpu().numpy()
        logits = np.array([np_vec[act.index()] for act in options])
        idx, prob = sample_softmax(logits)
        return {
            'options': options,
            'action': options[idx],
            'log_prob': math.log(prob),
            'value': values.item(),
            'state': new_state,
        }


def sample_softmax(logits):
    max_value = np.max(logits)
    probs = np.exp(logits - max_value)
    probs /= np.sum(probs)
    x = random.random()
    for i, y in enumerate(probs):
        x -= y
        if x <= 0:
            return i, probs[i]
    return len(logits) - 1, probs[-1]
