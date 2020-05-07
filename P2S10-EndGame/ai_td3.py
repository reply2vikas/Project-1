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
from torch.autograd import Variable
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

latent_dim = 128

# Creating the architecture of the Neural Network

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Actor(nn.Module):
    def __init__(self, action_dim, distance, orientation):
        super(Actor, self).__init__()

        self.encoder = torch.nn.ModuleList([  ## input size:[32, 32]
            torch.nn.Conv2d(1, 8, 3, 1, padding=1),  ## output size: [8, 32, 32]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 12, 3, 1, padding=1),  ## output size: [12, 32, 32]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(12),
            torch.nn.Conv2d(12, 16, 3, 2, padding=1),  ## output size: [16, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 24, 3, 1, padding=1),  ## output size: [24, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(24),
            torch.nn.Conv2d(24, 32, 3, 1, padding=1),  ## output size: [32, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 48, 3, 2, padding=1),  ## output size: [48, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(48),
            torch.nn.Conv2d(48, 64, 3, 1, padding=1),  ## output size: [64, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 3, 1, padding=1),  ## output size: [128, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.AdaptiveAvgPool2d((1,1)), ## output size: [128, 1, 1]
            Flatten(),  ## output: 128
        ])

        self.linear = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + 2, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, action_dim),
            torch.nn.Tanh(),
        ])

    def forward(self, x, distance, orientation):
        for layer in self.encoder:
            x = layer(x)
            # print('START')
            # print('x shape is ', x.shape)
            # print('distance shape is ', distance.shape)
            # print('orientation shape is ', orientation.shape)
        counter = 0
        for layer in self.linear:
            counter += 1
            if counter == 1:
                # print('BEFORE CONCAT')
                # print('x shape is ', x.shape)
                # print('distance shape is ', distance.shape)
                # print('orientation shape is ', orientation.shape)
                x = torch.cat([x, distance, orientation], 1)
                # print('AFTER CONCAT')
                # print('x shape is ', x.shape)
                # print('distance shape is ', distance.shape)
                # print('orientation shape is ', orientation.shape)
                x = layer(x)
            else:
                x = layer(x)
        return x


class Critic(nn.Module):
    def __init__(self, action_dim, distance, orientation):
        super(Critic, self).__init__()

        self.encoder_1 = torch.nn.ModuleList([  ## input size:[80, 80]
            torch.nn.Conv2d(1, 8, 3, 1, padding=1),  ## output size: [8, 32, 32]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 12, 3, 1, padding=1),  ## output size: [12, 32, 32]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(12),
            torch.nn.Conv2d(12, 16, 3, 2, padding=1),  ## output size: [16, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 24, 3, 1, padding=1),  ## output size: [24, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(24),
            torch.nn.Conv2d(24, 32, 3, 1, padding=1),  ## output size: [32, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 48, 3, 2, padding=1),  ## output size: [48, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(48),
            torch.nn.Conv2d(48, 64, 3, 1, padding=1),  ## output size: [64, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 3, 1, padding=1),  ## output size: [128, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.AdaptiveAvgPool2d((1,1)), ## output size: [128, 1, 1]
            Flatten(),  ## output: 128
        ])

        self.encoder_2 = torch.nn.ModuleList([  ## input size:[80, 80]
            torch.nn.Conv2d(1, 8, 3, 1, padding=1),  ## output size: [8, 32, 32]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 12, 3, 1, padding=1),  ## output size: [12, 32, 32]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(12),
            torch.nn.Conv2d(12, 16, 3, 2, padding=1),  ## output size: [16, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 24, 3, 1, padding=1),  ## output size: [24, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(24),
            torch.nn.Conv2d(24, 32, 3, 1, padding=1),  ## output size: [32, 16, 16]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 48, 3, 2, padding=1),  ## output size: [48, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(48),
            torch.nn.Conv2d(48, 64, 3, 1, padding=1),  ## output size: [64, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64, 128, 3, 1, padding=1),  ## output size: [128, 8, 8]
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.AdaptiveAvgPool2d((1,1)), ## output size: [128, 1, 1]
            Flatten(),  ## output: 128
        ])

        self.linear_1 = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim + 2, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
        ])

        self.linear_2 = torch.nn.ModuleList([
            torch.nn.Linear(latent_dim + action_dim + 2, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
        ])

    def forward(self, x, u, distance, orientation):
        x1 = x
        for layer in self.encoder_1:
            x1 = layer(x1)
        counter = 0
        # print('x1 shape is ',x1.size())
        # print('u shape is ',u.size())
        for layer in self.linear_1:
            counter += 1
            if counter == 1:
                x1 = torch.cat([x1, u, distance, orientation], 1)
                x1 = layer(x1)
            else:
                x1 = layer(x1)

        x2 = x
        for layer in self.encoder_2:
            x2 = layer(x2)
        counter = 0
        for layer in self.linear_2:
            counter += 1
            if counter == 1:
                x2 = torch.cat([x2, u, distance, orientation], 1)
                x2 = layer(x2)
            else:
                x2 = layer(x2)

        return x1, x2

    def Q1(self, x, u, distance, orientation):

        for layer in self.encoder_1:
            x = layer(x)

        counter = 0
        for layer in self.linear_1:
            counter += 1
            if counter == 1:
                x = torch.cat([x, u, distance, orientation], 1)
                x = layer(x)
            else:
                x = layer(x)

        return x


# Implementing TD3
class TD3(object):
  
  def __init__(self, action_dim, distance, orientation):

        self.action_dim = action_dim
        #self.distance = distance

        self.actor = Actor(action_dim, distance, orientation).to(device)
        self.actor_target = Actor(action_dim, distance, orientation).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.actor_loss = []

        self.critic = Critic(action_dim, distance, orientation).to(device)
        self.critic_target = Critic(action_dim, distance, orientation).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.critic_loss = []

        self.reward_window = []


  def select_action(self, state, distance, orientation):
    state = state.float().to(device)
    distance = distance.float().to(device)
    orientation = orientation.float().to(device)
    # print('shape of state is ', state.shape)
    # print('shape of distance is ', distance.view(-1,1).shape)
    # print('shape of orientation is ', orientation.view(-1,1).shape)
    # print('state is ', state)
    # print('distance is ', distance)
    # print('distance view is ', distance.view(-1,1))
    # print('orientation is ', orientation)
    # print('orientation view is ', orientation.view(-1,1))
    return self.actor(state, distance.view(-1,1), orientation.view(-1,1)).cpu().data.numpy().flatten()


  def train(self, replay_buffer, iterations=1000, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
    #print('IN TRAIN ', iterations)
    
    for it in range(iterations):
      # Sample replay buffer
      batch_states, batch_next_states, batch_distances, batch_next_distances, batch_orientations, batch_next_orientations, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
      state = torch.FloatTensor(batch_states).squeeze(1).to(device)
      #             print('state size: ' +str(state.size()))
      distance = torch.FloatTensor(batch_distances).squeeze(1).to(device)
      orientation = torch.FloatTensor(batch_orientations).squeeze(1).to(device)
      batch_actions = batch_actions.reshape((batch_size, self.action_dim))
      action = torch.FloatTensor(batch_actions).to(device)
      #             print('action size: ' +str(action.size()))
      next_state = torch.FloatTensor(batch_next_states).squeeze(1).to(device)
      #             print('next state size: ' +str(next_state.size()))
      batch_next_distances = batch_next_distances.reshape((batch_size, self.action_dim))
      next_distance = torch.FloatTensor(batch_next_distances).squeeze(1).to(device)
      batch_next_orientations = batch_next_orientations.reshape((batch_size, self.action_dim))
      next_orientation = torch.FloatTensor(batch_next_orientations).squeeze(1).to(device)
      done = torch.FloatTensor(1 - batch_dones).to(device)
      reward = torch.FloatTensor(batch_rewards).to(device)

      # Select action according to policy and add clipped noise
      noise = torch.FloatTensor(batch_actions).data.normal_(0, policy_noise).to(device)
      noise = noise.clamp(-noise_clip, noise_clip)
      next_action = (self.actor_target(next_state, next_distance.view(-1,1), orientation.view(-1,1)) + noise).clamp(-1, 1)
      
      # Step 7: The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      target_Q1, target_Q2 = self.critic_target(next_state, next_action, next_distance.view(-1,1), next_orientation.view(-1,1))
      
      # Step 8: We keep the minimum of these two Q-values: min(Qt1, Qt2)
      target_Q = torch.min(target_Q1, target_Q2)
      
      # Step 9: We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the discount factor
      target_Q = reward + ((1 - done) * discount * target_Q).detach()
      
      # Step 10: The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs
      current_Q1, current_Q2 = self.critic(state, action, distance.view(-1,1), orientation.view(-1,1))
      
      # Step 11: We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

      # Step 12: We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
      
      # Step 13: Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state, distance.view(-1,1), orientation.view(-1,1)), distance.view(-1,1), orientation.view(-1,1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Step 14: Still once every two iterations, we update the weights of the Actor target by polyak averaging
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # Step 15: Still once every two iterations, we update the weights of the Critic target by polyak averaging
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
  

  def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

  # Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
  
  # Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))