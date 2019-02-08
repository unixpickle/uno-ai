"""
Train an Uno agent.
"""

import argparse
import os
import random

import torch

from uno_ai.agent import Agent, BaselineAgent
from uno_ai.game import Game
from uno_ai.pool import Pool
from uno_ai.ppo import PPO
from uno_ai.rollouts import Rollout, RolloutBatch


def main():
    args = arg_parser().parse_args()
    device = torch.device(args.device)
    agent = Agent()
    agent.to(device)

    if os.path.exists(args.path):
        state_dict = torch.load(args.path, map_location=device)
        agent.load_state_dict(state_dict)

    pool = Pool(args.pool)
    if pool.empty():
        pool.add(agent)

    ppo = PPO(agent, epsilon=args.epsilon, lr=args.lr, ent_reg=args.entropy)
    while True:
        rollouts, mean_rew, mean_len = gather_rollouts(args, agent, pool)
        step_res = ppo.loop(rollouts, iters=args.iters)
        print('reward=%f steps=%f entropy_init=%f entropy_final=%f clipped=%f' %
              (mean_rew, mean_len, step_res[0]['entropy'], step_res[-1]['entropy'],
               step_res[-1]['clipped']))
        pool.add(agent)
        torch.save(agent.state_dict(), args.path)


def gather_rollouts(args, agent, pool):
    rollouts = []
    for _ in range(args.batch):
        if args.baseline:
            agents = [agent] + [BaselineAgent() for _ in range(args.players - 1)]
        else:
            agents = [agent] + [pool.sample(agent.device()) for _ in range(args.players - 1)]
        random.shuffle(agents)
        rs = Rollout.rollout(Game(args.players), agents)
        rollouts.append(rs[agents.index(agent)])
    mean_rew = sum(r.reward for r in rollouts) / len(rollouts)
    mean_len = sum(r.num_steps for r in rollouts) / len(rollouts)
    return RolloutBatch(rollouts, agent.device()), mean_rew, mean_len


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
    parser.add_argument('--baseline', help='train against a baseline', action='store_true')
    return parser


if __name__ == '__main__':
    main()
