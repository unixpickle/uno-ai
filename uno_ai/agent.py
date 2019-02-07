"""
Reinforcement Learning agents.
"""

import torch.nn as nn

from .actions import ACTION_VECTOR_SIZE
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
