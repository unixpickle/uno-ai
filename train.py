"""
Train an Uno agent.
"""

import argparse
import random

import torch

from uno_ai.agent import Agent
from uno_ai.game import Game
from uno_ai.pool import Pool
from uno_ai.ppo import PPO
from uno_ai.rollouts import Rollout, RolloutBatch


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)
    agent = Agent()
    agent.to(device)
    pool = Pool(args.pool)
    if pool.empty():
        pool.add(agent)
    ppo = PPO(agent, epsilon=args.epsilon, lr=args.lr, ent_reg=args.entropy)
    while True:
        rollouts, mean_rew = gather_rollouts(args, agent)
        step_res = ppo.loop(rollouts, iters=args.iters)
        print('reward=%f entropy_init=%f entropy_final=%f' %
              (mean_rew, step_res[0]['entropy'], step_res[-1]['entropy']))


def gather_rollouts(args, agent, pool):
    rollouts = []
    for _ in range(args.batch):
        agents = [agent] + [pool.sample() for _ in range(args.players - 1)]
        random.shuffle(agents)
        rs = Rollout.rollout(Game(args.players), agents)
        rollouts.append(rs[agents.index(agent)])
    return RolloutBatch(rollouts), sum(r.reward for r in rollouts) / len(rollouts)


def arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', help='path to latest model', default='model.pt')
    parser.add_argument('--pool', help='path to pool directory', default='agents')
    parser.add_argument('--lr', help='PPO learning rate', default=0.0001, type=float)
    parser.add_argument('--entropy', help='entropy bonus', default=0.01, type=float)
    parser.add_argument('--epsilon', help='PPO epsilon', default=0.2, type=float)
    parser.add_argument('--iters', help='PPO iterations', default=8, type=int)
    parser.add_argument('--players', help='number of players', default=4, type=int)
    parser.add_argument('--batch', help='rollouts per batch', default=128, type=int)
    parser.add_argument('--device', help='torch device to use', default='cpu')
    return parser


if __name__ == '__main__':
    main()
