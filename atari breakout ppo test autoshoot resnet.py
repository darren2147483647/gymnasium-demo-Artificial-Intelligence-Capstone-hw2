import torch
import torch.nn.functional as F
import gymnasium as gym
import gymnasium
import ale_py
import cv2
import numpy as np
from collections import deque
import time

import resnet34

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# ======= Wrapper classes (同 train 中定義) =======

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

# ======= Network 定義 (要與 train 相同) =======

import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = resnet34.resnet34(in_channel=4)
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(x)
        return self.actor(x), self.critic(x)

# ======= 測試主程式 =======
gymnasium.register_envs(ale_py)
if __name__ == '__main__':
    # env = gym.make("ALE/Breakout-v5", render_mode="human")  # 使用 human 模式顯示畫面
    # env = PreprocessAtariObs(env, shape=(84, 84))
    # env = FrameStack(env, k=4)

    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder='./video', episode_trigger=lambda e: True)
    env = PreprocessAtariObs(env, shape=(84, 84))
    env = FrameStack(env, k=4)

    n_actions = env.action_space.n - 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(n_actions).to(device)
    model.load_state_dict(torch.load("ppo_breakout_autoshoot_resnet_best.pth"))
    model.eval()

    for episode in range(5):  # 玩 5 局
        state, _ = env.reset()
        total_reward = 0
        no_reward_grace_period = 60
        no_reward_counter = 0
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(state_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()  # 選最大機率的動作
            
            if no_reward_counter > no_reward_grace_period and np.random.uniform() < 0.1:
                real_action = 1
                no_reward_counter = 0
            else:
                real_action = action if action == 0 else action + 1  # Adjust action to match the environment's action space
            next_state, reward, terminated, truncated, info = env.step(real_action)
            
            no_reward_counter = 0 if reward !=0 else (no_reward_counter + 1)

            state = next_state
            total_reward += reward

            time.sleep(0.01)  # 減慢遊戲速度以方便觀看

            if terminated or truncated:
                print(f"Episode {episode + 1} Reward: {total_reward}")
                break

    env.close()
