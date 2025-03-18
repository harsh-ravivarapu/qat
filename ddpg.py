import copy
import wandb
import torch
import torch.nn as nn


# https://github.com/Lizhi-sjtu/DRL-code-pytorch/blob/main/6.DDPG/DDPG.py


class DDPG:
    def __init__(self, actor, critic, device="cuda", batch_size=256, gamma=0.99, tau=0.005, actor_lr=0.001,
                 critic_lr=0.01):
        self.batch_size = batch_size  # batch size
        self.GAMMA = gamma  # discount factor
        self.TAU = tau  # Softly update the target network
        self.device = device

        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.to(device)
        self.actor_target.to(device)
        self.critic.to(device)
        self.critic_target.to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mse_loss = nn.MSELoss().to(self.device)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, a_logit = self.actor(s)
        # print("action:", a.size(), a_logit.size())
        a = a.data.cpu().numpy().flatten()
        a_logit = a_logit.data.cpu().numpy()
        return a, a_logit

    def learn(self, relay_buffer, wandb_run=None):
        batch_s, batch_a, batch_a_logit, batch_r, batch_s_, batch_dw = relay_buffer.sample(
            self.batch_size)  # Sample a batch
        batch_s, batch_a, batch_a_logit, batch_r, batch_s_, batch_dw = \
            batch_s.to(self.device), batch_a.to(self.device), batch_a_logit.to(self.device), batch_r.to(self.device), \
                batch_s_.to(self.device), batch_dw.to(self.device)
        # Compute the target Q
        with torch.no_grad():  # target_Q has no gradient
            Q_ = self.critic_target(batch_s_, self.actor_target(batch_s_)[1])
            target_Q = batch_r + self.GAMMA * (1 - batch_dw) * Q_

        # Compute the current Q and the critic loss
        current_Q = self.critic(batch_s, batch_a_logit)
        critic_loss = self.mse_loss(target_Q, current_Q)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute the actor loss
        actor_loss = - self.critic(batch_s, self.actor(batch_s)[1]).mean()
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Softly update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)

        if wandb_run:
            wandb_run.log({"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()})


def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s)  # We do not add noise when evaluating
            s_, r, done, _ = env.step(a)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


def reward_adapter(r, env_index):
    if env_index == 0:  # Pendulum-v1
        r = (r + 8) / 8
    elif env_index == 1:  # BipedalWalker-v3
        if r <= -100:
            r = -1
    return r
