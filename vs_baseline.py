"""
Play an agent against random agents.
"""

import argparse
import os
import random

import torch

from uno_ai.agent import Agent, BaselineAgent
from uno_ai.game import Game
from uno_ai.rollouts import Rollout


def main():
    args = arg_parser().parse_args()

    agent = Agent()
    if os.path.exists(args.path):
        state_dict = torch.load(args.path, map_location='cpu')
        agent.load_state_dict(state_dict)

    baseline = BaselineAgent()
    agents = [baseline] * (args.players - 1) + [agent]

    rewards = []
    while True:
        game = Game(args.players)
        random.shuffle(agents)
        rs = Rollout.rollout(game, agents)
        rewards.append(rs[agents.index(agent)].reward)
        print('mean=%f' % (sum(rewards) / len(rewards)))


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to latest model', default='model.pt')
    parser.add_argument('--players', help='number of players', default=4, type=int)
    return parser


if __name__ == '__main__':
    main()
