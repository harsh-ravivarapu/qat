import os
import gym
import matlab.engine
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization as quantization
import wandb

from ddpg import DDPG
from utils import seed_everything, init_weights_xavier

os.environ["WANDB_SILENT"] = "true"


class Brain(gym.Env):
    def __init__(self, freq, l, b, time_step, stride, window_size, n_obs, action_dim):
        self.freq = float(freq)
        self.len = float(l)
        self.b = float(b)
        self.isdone = False
        self.IT = None
        self.kk = None
        self.time_step = float(time_step)
        self.stride = float(stride)
        self.window_size = float(window_size)
        self.eng = matlab.engine.start_matlab()
        self.obs = None
        self.n_obs = n_obs
        self.action_dim = action_dim
        self.reward = None

    def reset(self):
        obs, IT = self.eng.reset_function_SMC_step(self.freq, self.len, self.time_step, self.stride, self.window_size, nargout=2)
        self.IT = float(IT)
        obs = np.array(obs)
        self.state = obs
        self.reward = 0
        return obs

    def step(self, action):
        action = np.array(action, dtype='float64')
        action = matlab.double(action)
        obs, reward, isdone, IT = self.eng.step_function_SMC_step(
            action, self.IT, self.freq, self.len, self.b, self.time_step, self.stride, self.window_size, nargout=4)
        obs = np.array(obs)
        self.IT = float(IT)
        reward = float(reward)
        isdone = float(isdone)
        info = {}
        self.state = obs
        self.reward += reward

        return obs, reward, isdone, info

    def random_action(self):
        return np.random.randint(2, size=self.n_obs)

    def uniform_action(self):
        action = np.zeros(self.n_obs)
        action[:: (self.n_obs // self.action_dim)] = 1
        return action

    def end(self):
        self.eng.quit()


class Actor(nn.Module):
    def __init__(self, state_dim=2, state_len=100, action_dim=2, shrink_dim=4):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.state_len = state_len
        self.action_dim = action_dim
        self.shrink_dim = shrink_dim

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.conv1 = nn.Conv1d(state_dim, 32, 3, padding=1)
        self.avg_pool1 = nn.AvgPool1d(shrink_dim)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.avg_pool2 = nn.AvgPool1d(shrink_dim)
        self.linear1 = nn.Linear(self.state_len // shrink_dim // shrink_dim * 64, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim * state_len)

    def forward(self, state):
        state = self.quant(state)
        output = self.avg_pool1(F.relu(self.conv1(state)))
        output = self.avg_pool2(F.relu(self.conv2(output)))
        output = output.view(-1, self.state_len // self.shrink_dim // self.shrink_dim * 64)
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        logits = self.linear3(output)

        output2 = logits.view(-1, self.action_dim, self.state_len)
        output2 = F.softmax(output2, dim=1)
        output2 = torch.argmax(output2, dim=-1)

        actions = torch.zeros(state.size(0), self.state_len).to(state.device)
        actions[torch.arange(state.size(0)).unsqueeze(1), output2] = 1

        actions = self.dequant(actions)
        return actions, logits


class Critic(nn.Module):
    def __init__(self, state_dim=2, state_len=100, action_dim=2, shrink_dim=4):
        super(Critic, self).__init__()
        self.state_len = state_len
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.shrink_dim = shrink_dim

        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        self.conv1 = nn.Conv1d(state_dim, 32, 3, padding=1)
        self.avg_pool1 = nn.AvgPool1d(shrink_dim)
        self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.avg_pool2 = nn.AvgPool1d(shrink_dim)
        self.linear1 = nn.Linear(self.state_len // shrink_dim // shrink_dim * 64, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_dim * state_len)
        self.linear4 = nn.Linear(action_dim * state_len, 256)
        self.linear5 = nn.Linear(256, 1)

    def forward(self, state, action_logits):
        state = self.quant(state)
        output = self.avg_pool1(F.relu(self.conv1(state)))
        output = self.avg_pool2(F.relu(self.conv2(output)))
        output = output.view(-1, self.state_len // self.shrink_dim // self.shrink_dim * 64)
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))

        action_output = F.relu(self.linear4(action_logits))
        all_output = output + action_output
        value = self.linear5(all_output)

        value = self.dequant(value)
        return value
    
class ReplayBuffer:
    def __init__(self, state_dim, state_len, action_dim, max_size):
        self.max_size = max_size
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim, state_len))
        self.a = np.zeros((self.max_size, state_len))
        self.a_logit = np.zeros((self.max_size, action_dim * state_len))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim, state_len))
        self.dw = np.zeros((self.max_size, 1))
        
    def store(self, s, a, a_logit, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.a_logit[self.count] = a_logit
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_a_logit = torch.tensor(self.a_logit[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)
        
        return batch_s, batch_a, batch_a_logit, batch_r, batch_s_, batch_dw
    


def main():
    seed = 0
    seed_everything(seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freq = 50  
    t_enviro = 1000  
    max_episodes = 10  
    steps = 30  
    critic_lr = 0.01  
    actor_lr = 0.01  
    gamma = 0.99  
    tau = 0.3  
    buffer_size = 2048  
    batch_size = 32  
    enviro_dt_factor = 1  
    enviro_stride = 1  
    enviro_window_size = 1000  
    state_dim = 1  
    step_size = 0.02 * enviro_dt_factor
    action_dim = freq * t_enviro // 1000
    n_obs = t_enviro // enviro_stride
    update_freq = 4
    start_steps = 10
    
    run = wandb.init(
        project="rl_brain", mode="offline",
        config={
            "critic_lr": critic_lr,
            "actor_lr": actor_lr,
            "seed": seed,
            "enviro_stride": enviro_stride,
            "enviro_window_size": enviro_window_size,
            "t_enviro": t_enviro,
            "freq": freq,
            "max_episodes": max_episodes,
            "steps": steps,
            "step_size": step_size,
            "buffer_size": buffer_size,
            "state_dim": state_dim,
            "batch_size": batch_size,
            "update_freq": update_freq
        })
    
    print("Total length:", t_enviro, "step_size", step_size)
    print('Length of observation vector: ', n_obs, 'Action Dim: ', action_dim)
    
    actor = Actor(state_dim=state_dim, state_len=n_obs, action_dim=action_dim, shrink_dim=2)
    actor.apply(init_weights_xavier)
    critic = Critic(state_dim=state_dim, state_len=n_obs, action_dim=action_dim, shrink_dim=2)
    critic.apply(init_weights_xavier)
    
    env = Brain(freq, t_enviro, action_dim, step_size, enviro_stride, enviro_window_size, n_obs, action_dim)
    agent = DDPG(actor, critic, batch_size=batch_size, gamma=gamma, tau=tau, actor_lr=actor_lr, critic_lr=critic_lr, device=device)
    replay_buffer = ReplayBuffer(state_dim=state_dim, action_dim=action_dim, state_len=n_obs, max_size=buffer_size)
    total_steps = 0
    
    for episode in range(max_episodes):
        print(f"Episode {episode}")
        s = env.reset()
        done = False
        beta_list = []
        ei_list = []
        r_list = []
        
        beta = s[0, :]
        ei = 0
        print(f"Step 0, beta: {np.mean(beta):.3f}, ei: {np.mean(ei):.3f}")
        wandb.log({"reward": 0, "beta": np.mean(beta), "ei": np.mean(ei), "episode": episode, "step": 0})
        
        for step in range(steps):
            if total_steps < start_steps:
                a = env.uniform_action()
                a_logit = np.zeros((action_dim * n_obs))
            else:
                a, a_logit = agent.choose_action(s)
                
            s_, r, done, _ = env.step(a)
            dw = True if done else False
            replay_buffer.store(s, a, a_logit, r, s_, dw)
            s = s_
            beta = s[0, :]
            ei = 0
            beta_list.append(np.mean(beta))
            ei_list.append(np.mean(ei))
            r_list.append(r)
            total_steps += 1
            
            for _ in range(update_freq):
                agent.learn(replay_buffer, wandb_run=run)
                
            print(f"Step {step}, reward: {r:.3f} beta: {np.mean(beta):.3f}, ei: {np.mean(ei):.3f}")
            wandb.log({"reward": r, "beta": np.mean(beta), "ei": np.mean(ei), "episode": episode, "step": step})
            
            if done:
                print("Start a new episode")
                break
            
        print(f"Episode {episode}, reward: {np.sum(r_list):.3f} beta: {np.mean(beta_list):.3f}, ei: {np.mean(ei_list):.3f}")
        wandb.log({"reward_epoch": np.sum(r_list), "beta_epoch": np.mean(beta_list), "ei_epoch": np.mean(ei_list), "episode": episode})
        
    #  Convert models to quantized format after training
    torch.quantization.convert(actor, inplace=True)
    torch.quantization.convert(critic, inplace=True)
    
    #  Save quantized models
    torch.save(actor.state_dict(), "quantized_actor.pth")
    torch.save(critic.state_dict(), "quantized_critic.pth")
    print("Quantized models saved successfully.")
    
    env.end()
    env.close()
    
    
if __name__ == "__main__":
    main()
    
