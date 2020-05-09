# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import pybullet_envs
import gym
from gym import wrappers
from collections import deque


class Actor(nn.Module):
  
  def __init__(self, state_dim, action_dim, max_action):
    super(Actor, self).__init__()
    self.conv1 = nn.Conv2d(1, 8, 3, 1,padding=1)
    self.bn1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 16, 3, 1,padding=1)
    self.bn2 = nn.BatchNorm2d(16)
    self.conv3 = nn.Conv2d(16, 16, 3, 1,padding=1)
    self.bn3 = nn.BatchNorm2d(16)
    self.dropout1 = nn.Dropout2d(0.2)
    self.dropout2 = nn.Dropout2d(0.2)
    # self.avgpooling = nn.AvgPool2d(20)
    self.fc1 = nn.Linear(6400+3, 128)
    self.fc2 = nn.Linear(128, action_dim)
    self.max_action = max_action

  def forward(self, state1, state2):
    x = F.relu(self.bn1(self.conv1(state1))) # 40x40x8
    x = F.relu(self.bn2(self.conv2(x))) # 40x40x16
    x = F.max_pool2d(x, 2) # 20x20x16
    x = self.dropout1(x) 

    x = F.relu(self.bn3(self.conv3(x))) # 20x20x16
    # x = self.avgpooling(x) #1x1x16
    x = torch.flatten(x, 1)
    x = torch.cat([x, state2], 1)

    x = self.dropout2(F.relu(self.fc1(x)))
    actions = self.fc2(x)
    return self.max_action * torch.tanh(actions)



class Critic(nn.Module):
  
  def __init__(self, state_dim, action_dim):
    super(Critic, self).__init__()
    # Defining the first Critic neural network
    self.conv1 = nn.Conv2d(1, 8, 3, 1,padding=1)
    self.conv2 = nn.Conv2d(8, 16, 3, 1,padding=1)
    self.conv3 = nn.Conv2d(16, 16, 3, 1,padding=1)
    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(16)
    self.bn3 = nn.BatchNorm2d(16)
    self.dropout1 = nn.Dropout2d(0.2)
    self.dropout2 = nn.Dropout2d(0.2)
    self.avgpooling1 = nn.AvgPool2d(20)
    self.fc1 = nn.Linear(6400+4, 128)
    self.fc2 = nn.Linear(128, 1)
    # Defining the second Critic neural network
    self.conv4 = nn.Conv2d(1, 8, 3, 1,padding=1)
    self.conv5 = nn.Conv2d(8, 16, 3, 1,padding=1)
    self.conv6 = nn.Conv2d(16, 16, 3, 1,padding=1)
    self.bn4 = nn.BatchNorm2d(8)
    self.bn5 = nn.BatchNorm2d(16)
    self.bn6 = nn.BatchNorm2d(16)
    self.dropout3 = nn.Dropout2d(0.2)
    self.dropout4 = nn.Dropout2d(0.2)
    self.avgpooling2 = nn.AvgPool2d(20)
    self.fc3 = nn.Linear(6400+4, 128)
    self.fc4 = nn.Linear(128, 1)

  def forward(self, state1, state2, u):
    # xu = torch.cat([x, u], 1)
    # Forward-Propagation on the first Critic Neural Network
    x1 = F.relu(self.bn1(self.conv1(state1))) # 38x38x8
    x1 = F.relu(self.bn2(self.conv2(x1))) # 36x36x16
    x1 = F.max_pool2d(x1, 2) # 18x18x16
    x1 = self.dropout1(x1) 
    x1 = F.relu(self.bn3(self.conv3(x1))) # 16x16x16
    # x1 = self.avgpooling1(x1) #1x1x16
    x1 = torch.flatten(x1, 1)
    x1 = torch.cat([x1,state2, u], 1)
    x1 = F.relu(self.fc1(x1))
    x1 = self.dropout2(x1)
    x1 = self.fc2(x1)

    # Forward-Propagation on the second Critic Neural Network
    x2 = F.relu(self.bn4(self.conv4(state1))) # 38x38x8
    x2 = F.relu(self.bn5(self.conv5(x2))) # 36x36x16
    x2 = F.max_pool2d(x2, 2) # 18x18x16
    x2 = self.dropout3(x2) 
    x2 = F.relu(self.bn6(self.conv6(x2))) # 16x16x16
    # x2 = self.avgpooling2(x2) #1x1x16
    x2 = torch.flatten(x2, 1)
    x2 = torch.cat([x2, state2, u], 1)
    x2 = F.relu(self.fc3(x2))
    x2 = self.dropout4(x2)
    x2 = self.fc4(x2)
    return x1, x2

  def Q1(self, state1, state2, u):
    x1 = F.relu(self.bn1(self.conv1(state1))) # 38x38x8
    x1 = F.relu(self.bn2(self.conv2(x1))) # 36x36x16
    x1 = F.max_pool2d(x1, 2) # 18x18x16
    x1 = self.dropout1(x1) 
    x1 = F.relu(self.bn3(self.conv3(x1))) # 16x16x16
    # x1 = self.avgpooling1(x1) #1x1x16
    x1 = torch.flatten(x1, 1)
    x1 = torch.cat([x1,state2, u], 1)
    x1 = F.relu(self.fc1(x1))
    x1 = self.dropout2(x1)
    x1 = self.fc2(x1)
    return x1

class ReplayBuffer(object):

  def __init__(self, max_size=1e5):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0

  def add(self, transition):
    if len(self.storage) == self.max_size:
      self.storage[int(self.ptr)] = transition
      self.ptr = (self.ptr + 1) % self.max_size
    else:
      self.storage.append(transition)

  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states1, batch_states2, batch_next_states1, batch_next_states2, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], []
    for i in ind: 
      state1, state2, next_state1, next_state2, action, reward, done = self.storage[i]
      batch_states1.append(np.array(state1, copy=False))
      batch_states2.append(np.array(state2, copy=False))
      batch_next_states1.append(np.array(next_state1, copy=False))
      batch_next_states2.append(np.array(next_state2, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states1), np.array(batch_states2), np.array(batch_next_states1), np.array(batch_next_states2), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)


# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class
class TD3(object):
  
  def __init__(self, state_dim, action_dim, max_action):
    self.actor = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = 0.007,weight_decay=0.0005)
    self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.actor_optimizer, 'min')
    self.critic = Critic(state_dim, action_dim).to(device)
    self.critic_target = Critic(state_dim, action_dim).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = 0.007,weight_decay=0.0005)
    self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_optimizer, 'min')

    #########################################################
    model_parameters = filter(lambda p: p.requires_grad, self.actor.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of actor params",params)
    model_parameters = filter(lambda p: p.requires_grad, self.critic.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("number of critic params",params)
    ################################################
    
    self.max_action = max_action
    # self.gamma = gamma
    self.reward_window = []
    # self.model = Network(input_size, nb_action)
    self.memory = ReplayBuffer()
    self.last_state1 = torch.zeros(state_dim)#.unsqueeze(0)
    self.last_state2 = torch.Tensor(3)#.unsqueeze(0)
    self.last_action = 0
    self.last_reward = 0
    self.action_dim = action_dim
    self.state_dim = state_dim
    # self.episode_timesteps = 0
    self.episode_reward = 0
    self.episode_num = 0
    

  def select_action(self, state1, state2):
    state1 = torch.Tensor(state1.unsqueeze(0)).to(device)
    state2 = torch.Tensor(state2.unsqueeze(0)).to(device)
    return self.actor(state1, state2).cpu().data.numpy().flatten()

  def train(self, replay_buffer, iterations ,batch_size=30, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    
    # self.critic_scheduler.step(critic_loss)
    # self.actor_scheduler.step(actor_loss)
    for it in range(iterations):
      
      # Step 4: We sample a batch of transitions (s, s’, a, r) from the memory
      batch_states1,batch_states2, batch_next_states1,batch_next_states2, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)

      state1 = torch.Tensor(batch_states1).to(device)
      state2 = torch.Tensor(batch_states2).to(device)
      next_state1 = torch.Tensor(batch_next_states1).to(device)
      next_state2 = torch.Tensor(batch_next_states2).to(device)
      action = torch.Tensor(batch_actions).to(device).unsqueeze(1)
      reward = torch.Tensor(batch_rewards).to(device)
      dones = torch.Tensor(batch_dones).to(device)
      
      # Step 5: From the next state s’, the Actor target plays the next action a’
      next_action = self.actor_target(next_state1,next_state2)
      
      # Step 6: We add Gaussian noise to this next action a’ and we clamp it in a range of values supported by the environment
      noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip).unsqueeze(1)
      next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      
      target_Q1, target_Q2 = self.critic_target(next_state1,next_state2, next_action)
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - dones) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state1, state2, action)
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
      if it % 20 == 0:
        print("_________________________")
        print("iteration number",it)
        print("critic loss",critic_loss)
      
      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      # self.critic_scheduler.step()
      # self.critic_scheduler.step(critic_loss)
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state1, state2, self.actor(state1, state2)).mean()
        if it % 20 == 0:
          print("actor loss",actor_loss)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # self.actor_scheduler.step()
        # self.actor_scheduler.step(actor_loss)
        self.actor_optimizer.step()
        # if it % 2*policy_freq == 0:
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
          
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  
  def update(self, reward, new_signal1,new_signal2, Done,count,episode_timesteps):
        # import pdb; pdb.set_trace()
        new_state1 = torch.Tensor(new_signal1).float().unsqueeze(0)
        new_state2 = torch.Tensor(new_signal2).float()#.unsqueeze(0)
        self.memory.add((self.last_state1, self.last_state2, new_state1, new_state2, self.last_action, self.last_reward, Done))

        
        if Done:
          print("num of steps for current episode", episode_timesteps)
          print("reward {}, for episode {} ------ ".format(self.episode_reward,self.episode_num))
          print("steps so far: ",count)
          batch_size, discount, tau, policy_noise, noise_clip, policy_freq = 100, 0.99,0.005,1,2.5,2
          start = time.time()
          self.train(self.memory, episode_timesteps,batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
          print("time for the episode", time.time()-start)
          
          self.episode_reward = 0
          episode_timesteps = 0
          self.episode_num += 1

        # Initial random actions
        if count < 2500:
          action = random.uniform(-self.max_action,self.max_action)
        else:
          action = self.select_action(new_state1, new_state2)
          print("action---- from policy", action)
          expl_noise = 0.5
          # Noise added for the purpose of exloration
          if expl_noise != 0:
            action = (action + np.random.normal(0, expl_noise, size=1)).clip(-self.max_action, self.max_action)
          action = action[0]
        self.last_action = action
        self.last_state1 = new_state1
        self.last_state2 = new_state2
        self.last_reward = reward

        self.episode_reward += reward

        episode_timesteps += 1

        if count %  4000 == 0:
          if not os.path.exists("./pytorch_models"):
            os.makedirs("./pytorch_models")
          self.save()

        return action, episode_timesteps

  def save(self):
    torch.save(self.actor.state_dict(), 'actor.pth')
    torch.save(self.critic.state_dict(), 'critic.pth')
    
  def load(self):
    self.actor.load_state_dict(torch.load('actor.pth'))
    self.critic.load_state_dict(torch.load('critic.pth'))
