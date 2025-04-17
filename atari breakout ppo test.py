import torch
import torch.nn.functional as F
import gymnasium as gym
import gymnasium
import ale_py
import cv2
import numpy as np
from collections import deque
import time
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

# ======= 測試主程式 =======
gymnasium.register_envs(ale_py)
if __name__ == '__main__':
    env = gym.make("ALE/Breakout-v5", render_mode="human")  # 使用 human 模式顯示畫面
    env = PreprocessAtariObs(env, shape=(84, 84))
    env = FrameStack(env, k=4)

    # env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, video_folder='./video', episode_trigger=lambda e: True)
    # env = PreprocessAtariObs(env, shape=(84, 84))
    # env = FrameStack(env, k=4)

    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ActorCritic(n_actions).to(device)
    model.load_state_dict(torch.load("ppo_breakout.pth", map_location=device))
    model.eval()

    for episode in range(5):  # 玩 5 局
        state, _ = env.reset()
        total_reward = 0
        live_now = 5
        while True:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _ = model(state_tensor)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()  # 選最大機率的動作

            next_state, reward, terminated, truncated, info = env.step(action)

            live_num=info.get("lives")
            if(live_num!=live_now):
                action=1
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                live_now=live_num

            state = next_state
            total_reward += reward

            time.sleep(0.01)  # 減慢遊戲速度以方便觀看

            if terminated or truncated:
                print(f"Episode {episode + 1} Reward: {total_reward}")
                break

    env.close()
