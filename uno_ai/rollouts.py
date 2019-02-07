"""
Gather trajectories from agents on a game.
"""


class Rollout:
    def __init__(self, observations, outputs, reward):
        self.observations = observations
        self.outputs = outputs
        self.reward = reward

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
        for i, r in enumerate(rollouts):
            if i == game.winner():
                r.reward = 1.0
        return rollouts
