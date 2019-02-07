"""
Pools of agents.
"""

import os
import random

import torch

from .agent import Agent


class Pool:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        self.agent_names = [x for x in os.listdir(dir_path) if x.endswith('.pt')]

    def empty(self):
        return len(self.agent_names) == 0

    def add(self, agent):
        """
        Add an agent to the pool.
        """
        idx = 0
        while ('%d.pt' % idx) in self.agent_names:
            idx += 1
        name = '%d.pt' % idx
        torch.save(agent.state_dict(), os.path.join(self.dir_path, name))
        self.agent_names.append(name)

    def sample(self, device):
        """
        Sample an agent from the pool.
        """
        name = random.choice(self.agent_names)
        state_dict = torch.load(os.path.join(self.dir_path, name), map_location=device)
        res = Agent()
        res.to(device)
        res.load_state_dict(state_dict)
        return res
