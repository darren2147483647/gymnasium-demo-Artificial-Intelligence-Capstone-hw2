import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.


# Basic Q-netowrk
class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden*2)
        self.out = nn.Linear(n_hidden*2, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


# Deep Q-Network, composed of one eval network, one target network
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, batch_size, lr, epsilon, gamma, target_replace_iter, memory_capacity, device="cpu"):
        self.eval_net, self.target_net = Net(n_states, n_actions, n_hidden), Net(n_states, n_actions, n_hidden)
        self.eval_net, self.target_net = self.eval_net.to(device), self.target_net.to(device)
        self.memory = np.zeros((memory_capacity, n_states * 2 + 3)) # initialize memory, each memory slot is of size (state + next state + reward + action + done)
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
        # Pack the experience
        transition = np.hstack((state, [action, reward, done], next_state))

        # Replace the old memory with new memory
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # Randomly select a batch of memory to learn from
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_state = torch.FloatTensor(b_memory[:, :self.n_states]).to(self.device)
        b_action = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)).to(self.device)
        b_reward = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(self.device)
        b_next_state = torch.FloatTensor(b_memory[:, -self.n_states:]).to(self.device)
        b_done = torch.BoolTensor(b_memory[:, self.n_states+2:self.n_states+3]).to(self.device)

        # Compute loss between Q values of eval net & target net
        q_eval = self.eval_net(b_state).gather(1, b_action) # evaluate the Q values of the experiences, given the states & actions taken at that time
        q_next = self.target_net(b_next_state).detach() # detach from graph, don't backpropagate
        q_target = (b_reward + self.gamma * q_next.max(1)[0].view(self.batch_size, 1))*(~b_done) # compute the target Q values
        loss = self.loss_func(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network every few iterations (target_replace_iter), i.e. replace target net with eval net
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # Environment parameters
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    # Hyper parameters
    n_hidden = 50
    batch_size = 32
    lr = 0.01                 # learning rate
    epsilon = 0.1             # epsilon-greedy, factor to explore randomly
    gamma = 0.9               # reward discount factor
    target_replace_iter = 100 # target network update frequency
    memory_capacity = 10000
    n_episodes = 1000000

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
        state, _ = env.reset() # reset environment to initial state for each episode
        while True:
            env.render()

            # Agent takes action
            action = dqn.choose_action(state) # choose an action based on DQN
            next_state, reward, done, _, info = env.step(action) # do the action, get the reward
            done = done or _

            # Keep the experience in memory
            dqn.store_transition(state, action, reward, next_state, done)

            # Accumulate reward
            rewards += reward

            # If enough memory stored, agent learns from them via Q-learning
            if dqn.memory_counter > memory_capacity:
                dqn.learn()

            # Transition to next state
            state = next_state

            if done:
                if rewards > best_reward:
                    best_reward = rewards
                    torch.save({"eval_net":dqn.eval_net.state_dict(),"target_net":dqn.target_net.state_dict()}, 'dqn_cartpole_best.pth') # save the model
                moving_reward = moving_reward * 0.9 + rewards * 0.1 # exponential moving average of the reward
                if (i_episode+1)%100 == 0:
                    print('Episode {} finished after {} timesteps, moving of total rewards {}'.format(i_episode+1,t+1, moving_reward))
                if (i_episode+1)%1000 == 0:
                    torch.save({"eval_net":dqn.eval_net.state_dict(),"target_net":dqn.target_net.state_dict()}, 'dqn_cartpole_{}.pth'.format(i_episode+1))
                #print('Episode {} finished after {} timesteps, total rewards {}'.format(i_episode+1,t+1, rewards))
                break

            t += 1
        reward_history.append(rewards)
        if sum(reward_history[-10:])/10 >= 200:
            print("Training finished at episode {}!".format(i_episode+1))
            break

    torch.save({"eval_net":dqn.eval_net.state_dict(),"target_net":dqn.target_net.state_dict()}, 'dqn_cartpole.pth') # save the model

    env.close()

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DQN Training Reward History')
    plt.legend()
    plt.grid(True)
    plt.show()
