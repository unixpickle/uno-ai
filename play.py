"""
Play against an Uno agent.
"""

import argparse
import os

import torch

from uno_ai.actions import NopAction
from uno_ai.agent import Agent
from uno_ai.game import Game
from uno_ai.rollouts import Rollout


def main():
    args = arg_parser().parse_args()

    agent = Agent()
    if os.path.exists(args.path):
        state_dict = torch.load(args.path, map_location='cpu')
        agent.load_state_dict(state_dict)

    game = Game(args.players)
    agents = [HumanAgent()] + [agent] * (args.players - 1)
    Rollout.rollout(game, agents)


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to latest model', default='model.pt')
    parser.add_argument('--players', help='number of players', default=4, type=int)
    return parser


class HumanAgent:
    def step(self, game, player, state):
        print('----------------------------')
        print('Discard:', game.discard())
        print('')
        print('Cards:')
        for i, card in enumerate(game.hands()[player]):
            print('%d. %s' % (i, card))
        print('')
        if game.turn() != player:
            input('Hit enter to continue.')
            return {
                'action': NopAction(),
                'state': None
            }
        else:
            print('Options:')
            for i, option in enumerate(game.options()):
                print('%d. %s' % (i, option))
            idx = input('Choose option: ')
            if idx == '':
                idx = '0'
            return {
                'action': game.options()[int(idx)],
                'state': None,
            }


if __name__ == '__main__':
    main()
