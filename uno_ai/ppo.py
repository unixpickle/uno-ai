"""
Proximal Policy Optimization.
"""

import torch
import torch.functional as F
import torch.optim as optim


class PPO:
    def __init__(self, agent, epsilon=0.2, lr=1e-4, ent_reg=1e-2):
        self.agent = agent
        self.epsilon = epsilon
        self.optim = optim.Adam(agent.parameters(), lr=lr)
        self.ent_reg = ent_reg

    def loop(self, batch, iters=8):
        steps = []
        for _ in range(iters):
            steps.append(self.step(batch))
        return steps

    def step(self, batch):
        logits, values, _ = self.agent(batch.observations)
        masked_logits = logits - (1 - batch.masks) * 10000
        all_probs = F.log_softmax(masked_logits, dim=-1)
        log_probs = torch.sum(all_probs * batch.actions, dim=-1)

        vf_loss = torch.mean(torch.pow(values - batch.targets, 2))

        ratio = torch.exp(log_probs - batch.log_probs)
        clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        pi_loss = -torch.mean(torch.min(ratio * batch.advs, clip_ratio * batch.advs))

        neg_entropy = torch.sum(torch.exp(log_probs) * log_probs, dim=-1)
        ent_loss = torch.mean(self.ent_reg * neg_entropy)

        loss = vf_loss + pi_loss + ent_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return {
            'vf_loss': vf_loss.item(),
            'pi_loss': pi_loss.item(),
            'ent_loss': ent_loss.item(),
            'entropy': -neg_entropy.item(),
        }
