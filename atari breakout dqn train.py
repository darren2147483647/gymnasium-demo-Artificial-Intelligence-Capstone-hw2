import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium
import ale_py
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

import gymnasium as gym
import numpy as np
import cv2
from collections import deque

class PreprocessAtariObs(gym.ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(1, *shape), dtype=np.float32
        )

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

        shp = env.observation_space.shape  # (1, 84, 84)
        self.observation_space = gym.spaces.Box(
            low=0, high=1.0, shape=(k, *shp[1:]), dtype=np.float32
        )

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


class NoNoopWrapper(gymnasium.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.valid_actions = [1, 2, 3]  # 排除 0 (NOOP)
        self.action_space = gymnasium.spaces.Discrete(len(self.valid_actions))

    def action(self, action):
        return self.valid_actions[action]


class Net(nn.Module):
    def __init__(self, n_actions):
        super(Net, self).__init__()
        '''
        n_actions: 動作數量 (如 4)
        輸入 shape: (batch, 4, 84, 84)
        '''
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)  # -> (batch, 32, 20, 20)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)                          # -> (batch, 64, 9, 9)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)                          # -> (batch, 64, 7, 7)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        # x: (batch, 4, 84, 84)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        actions_value = self.out(x)
        return actions_value



# Deep Q-Network, composed of one eval network, one target network
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity, device="cpu"):
        self.eval_net, self.target_net = Net(n_actions), Net(n_actions)
        self.eval_net, self.target_net = self.eval_net.to(device), self.target_net.to(device)
        self.memory_action = np.zeros((memory_capacity, 1)) # initialize memory for actions
        self.memory_reward = np.zeros((memory_capacity, 1)) # initialize memory for rewards
        self.memory_state = np.zeros((memory_capacity, n_states[0], n_states[1], n_states[2])) # initialize memory for states
        self.memory_next_state = np.zeros((memory_capacity, n_states[0], n_states[1], n_states[2])) # initialize memory for next states
        self.memory_done = np.zeros((memory_capacity, 1)) # initialize memory for done
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        self.memory_counter = 0
        self.learn_step_counter = 0 # for target network update

        self.n_states = n_states
        self.n_actions = n_actions
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
        self.device = device

        self.epsilon_start = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999
        self.epsilon = self.epsilon_start

    def choose_action(self, state):
        x = torch.unsqueeze(torch.FloatTensor(state), 0)

        # epsilon-greedy
        if np.random.uniform() < self.epsilon: # random
            action = np.random.randint(0, self.n_actions)
        else: # greedy
            actions_value = self.eval_net(x.to(self.device)).to("cpu") # feed into eval net, get scores for each action
            action = torch.max(actions_value, 1)[1].data.numpy()[0] # choose the one with the largest score

        return action

    def store_transition(self, state, action, reward, next_state, done):

        # Replace the old memory with new memory
        if self.memory_counter <= self.memory_capacity:
            index=self.memory_counter % self.memory_capacity
        else:
            index=np.random.randint(self.memory_capacity)
        self.memory_action[index, :] = action
        self.memory_reward[index, :] = reward
        self.memory_state[index, :] = state
        self.memory_next_state[index, :] = next_state
        self.memory_done[index, :] = done
        self.memory_counter += 1

    def learn(self):
        # Randomly select a batch of memory to learn from
        if self.memory_counter > self.memory_capacity:
            sample_index=np.random.choice(self.memory_capacity,size=self.batch_size)
        else:
            sample_index=np.random.choice(self.memory_counter,size=self.batch_size)

        b_state = torch.FloatTensor(self.memory_state[sample_index, :]).to(device)
        b_action = torch.LongTensor(self.memory_action[sample_index, :].astype(int)).to(device)
        b_reward = torch.FloatTensor(self.memory_reward[sample_index, :]).to(device)
        b_next_state = torch.FloatTensor(self.memory_next_state[sample_index, :]).to(device)
        b_done = torch.BoolTensor(self.memory_done[sample_index, :]).to(device)

        # Compute loss between Q values of eval net & target net
        q_eval = self.eval_net(b_state).gather(1, b_action) # evaluate the Q values of the experiences, given the states & actions taken at that time
        q_next = self.target_net(b_next_state).detach() # detach from graph, don't backpropagate
        q_target = b_reward + (self.gamma * q_next.max(1)[0].view(self.batch_size, 1))*(~b_done) # compute the target Q values
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network every few iterations (target_replace_iter), i.e. replace target net with eval net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

gymnasium.register_envs(ale_py)

if __name__ == '__main__':
    #env = gymnasium.make("ALE/Casino-v5", render_mode="human", mode=2)
    #env = gymnasium.make("ALE/Casino-v5", render_mode="rgb_array", mode=2)

    #env = NoNoopWrapper(env)
    env = gymnasium.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = PreprocessAtariObs(env, shape=(84, 84))
    env = FrameStack(env, k=4)

    
    # Environment parameters
    n_actions = env.action_space.n
    n_states = env.observation_space.shape

    # Hyper parameters
    n_hidden = 50
    batch_size = 32
    lr = 0.001                 # learning rate
    epsilon = 0.1             # epsilon-greedy, factor to explore randomly
    gamma = 0.99               # reward discount factor
    target_replace_iter = 5000 # target network update frequency
    memory_capacity = 10000
    n_episodes = 100000000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DQN
    dqn = DQN(n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity, device)

    reward_history=[] # for plotting the reward history
    best_reward = 0
    moving_reward = 0

    # Collect experience
    for i_episode in range(n_episodes):
        t = 0 # timestep
        rewards = 0 # accumulate rewards for each episode
        state, info = env.reset() # reset environment to initial state for each episode
        live_now=5
        while True:
            env.render()

            dqn.update_epsilon() # update epsilon for exploration
            # Agent takes action
            action = dqn.choose_action(state) # choose an action based on DQN
            next_state, reward, terminated, truncated, info = env.step(action) # do the action, get the reward
            done = terminated or truncated
            live_num=info.get("lives")
            if(live_num!=live_now):
                action=1
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                live_now=live_num

            # Keep the experience in memory
            dqn.store_transition(state, action, reward, next_state, done)

            # Accumulate reward
            rewards += reward

            # If enough memory stored, agent learns from them via Q-learning
            if dqn.memory_counter > batch_size*4:
                dqn.learn()

            # Transition to next state
            state = next_state
            
            if done:
                if rewards > best_reward:
                    best_reward = rewards
                    torch.save({"eval_net":dqn.eval_net.state_dict(),"target_net":dqn.target_net.state_dict()}, 'atari_dqn_breakout_best.pth')
                moving_reward = moving_reward * 0.9 + rewards * 0.1
                if (i_episode+1) % 10 == 0:
                    print('Episode {} finished after {} timesteps, moving of total rewards {}'.format(i_episode+1,t+1, moving_reward))
                if (i_episode+1) % 100 == 0:
                    torch.save({"eval_net":dqn.eval_net.state_dict(),"target_net":dqn.target_net.state_dict()}, 'atari_dqn_breakout_{}.pth'.format(i_episode+1))
                #print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode+1,t+1, rewards))
                break

            t += 1
        reward_history.append(rewards)
        if sum(reward_history[-10:])/10 >= 1000:
            print("Training finished at episode {}!".format(i_episode+1))
            break

    torch.save({"eval_net":dqn.eval_net.state_dict(),"target_net":dqn.target_net.state_dict()}, 'atari_dqn_breakout.pth') # save the model
    env.close()
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Reward History')
    plt.legend()
    plt.grid(True)
    plt.show()


