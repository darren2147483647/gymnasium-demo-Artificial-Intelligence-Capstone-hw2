import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import gymnasium
import ale_py
import cv2
from collections import deque
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ========= Env Wrappers =========

class PreprocessAtariObs(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(1, *shape), dtype=np.float32)

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        obs = obs.astype(np.float32) / 255.0
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(k, *shp[1:]), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        return np.array(self.frames)

# ========= Actor-Critic Network =========

class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # (batch, 32, 20, 20)
        x = F.relu(self.conv2(x))  # (batch, 64, 9, 9)
        x = F.relu(self.conv3(x))  # (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return self.actor(x), self.critic(x)

# ========= PPO Agent =========

class PPO:
    def __init__(self, n_actions, device):
        self.device = device
        self.n_actions = n_actions
        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.1
        self.K_epochs = 4
        self.batch_size = 64
        self.lr = 2.5e-4

        self.policy = ActorCritic(n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, _ = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), dist.entropy().item()

    def compute_gae(self, rewards, masks, values):
        values = values + [0]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lam * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def update(self, memory):
        states = torch.FloatTensor(np.array(memory['states'])).to(self.device)
        actions = torch.LongTensor(memory['actions']).to(self.device)
        log_probs_old = torch.FloatTensor(memory['log_probs']).to(self.device)
        returns = torch.FloatTensor(memory['returns']).to(self.device)
        advantages = returns - torch.FloatTensor(memory['values']).to(self.device)

        for _ in range(self.K_epochs):
            for i in range(0, len(states), self.batch_size):
                idx = slice(i, i + self.batch_size)
                logits, value = self.policy(states[idx])
                dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
                log_probs = dist.log_prob(actions[idx])
                ratios = torch.exp(log_probs - log_probs_old[idx])

                surr1 = ratios * advantages[idx]
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value.squeeze(), returns[idx])
                loss = actor_loss + 0.5 * critic_loss - 0.01 * dist.entropy().mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

# ========= Main =========
gymnasium.register_envs(ale_py)
if __name__ == '__main__':
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = PreprocessAtariObs(env, shape=(84, 84))
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPO(n_actions, device)

    max_episodes = 10000
    max_timesteps = 5000
    update_timestep = 2048

    timestep = 0
    reward_history = []

    memory = {'states': [], 'actions': [], 'log_probs': [], 'rewards': [], 'dones': [], 'values': []}

    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0

        for t in range(max_timesteps):
            timestep += 1
            action, log_prob, _ = agent.select_action(state)
            logits, value = agent.policy(torch.FloatTensor(state).unsqueeze(0).to(device))
            value = value.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['log_probs'].append(log_prob)
            memory['rewards'].append(reward)
            memory['dones'].append(1 - int(done))
            memory['values'].append(value)

            state = next_state
            total_reward += reward

            if timestep % update_timestep == 0 or done:
                with torch.no_grad():
                    _, last_value = agent.policy(torch.FloatTensor(state).unsqueeze(0).to(device))
                returns = agent.compute_gae(memory['rewards'], memory['dones'], memory['values'] + [last_value.item()])
                memory['returns'] = returns
                agent.update(memory)
                memory = {k: [] for k in memory}
                timestep = 0

            if done:
                reward_history.append(total_reward)
                if (episode + 1) % 10 == 0:
                    avg_reward = np.mean(reward_history[-10:])
                    print(f"Episode {episode + 1}, Reward: {total_reward}, Avg: {avg_reward}")
                break

    env.close()
    torch.save(agent.policy.state_dict(), "ppo_breakout.pth")
    plt.plot(reward_history)
    plt.title("PPO Breakout Training")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()
