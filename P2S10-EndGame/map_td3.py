# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
#import matplotlib.pyplot as plt
import time
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from PIL import ImageDraw
from kivy.graphics.texture import Texture

# Importing the Dqn object from our AI in ai.py
from ai_td3 import TD3

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1350')
Config.set('graphics', 'height', '550')

# Implementing Experience Replay
class ReplayBuffer(object):

  def __init__(self, max_size=1e6):
    self.storage = []
    self.max_size = max_size
    self.ptr = 0


  def add(self, state, new_state, distance, new_distance, orientation, new_orientation, action, reward, done_bool):
    data = (state, new_state, distance, new_distance, orientation, new_orientation, action, reward, done_bool)
    if len(self.storage) == self.max_size:
        self.storage[int(self.ptr)] = data
        self.ptr = (self.ptr + 1) % self.max_size
    else:
        self.storage.append(data)

  
  def sample(self, batch_size):
    ind = np.random.randint(0, len(self.storage), size=batch_size)
    batch_states, batch_next_states, batch_distances, batch_next_distances, batch_orientations, batch_next_orientations, batch_actions, batch_rewards, batch_dones = [], [], [], [], [], [], [], [], []
    for i in ind:
      state, next_state, distance, next_distance, orientation, next_orientation, action, reward, done = self.storage[i]
      batch_states.append(np.array(state, copy=False))
      batch_next_states.append(np.array(next_state, copy=False))
      batch_distances.append(np.array(distance, copy=False))
      batch_next_distances.append(np.array(next_distance, copy=False))
      batch_orientations.append(np.array(orientation, copy=False))
      batch_next_orientations.append(np.array(next_orientation, copy=False))
      batch_actions.append(np.array(action, copy=False))
      batch_rewards.append(np.array(reward, copy=False))
      batch_dones.append(np.array(done, copy=False))
    return np.array(batch_states), np.array(batch_next_states), np.array(batch_distances).reshape(-1, 1), np.array(batch_next_distances).reshape(-1, 1), np.array(batch_orientations).reshape(-1, 1), np.array(batch_next_orientations).reshape(-1, 1), np.array(batch_actions), np.array(batch_rewards).reshape(-1, 1), np.array(batch_dones).reshape(-1, 1)



# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

seed = 0
save_models = True # Boolean checker whether or not to save the pre-trained model
torch.manual_seed(seed)
np.random.seed(seed)
action_dim = 1
max_action = 5

_max_episode_steps = 500
_max_eval_episodes = 20
episode_num = 0

# Initializing the last distance
distance = 0
orientation = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = TD3(action_dim, distance, orientation)
replay_buffer = ReplayBuffer()
scores = []
im = CoreImage("./images/MASK1.png")

file_name = "%s_%s" % ("TD3", str(seed))
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

if not os.path.exists("./results"):
    os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")


# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur,largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1350
    goal_y = 550
    first_update = False
    global swap
    swap = 0


# Creating the car class

class Car(Widget):
    
    angle = NumericProperty(0)
    rotation = float(0) #NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)


    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation

        # xx = goal_x - self.x
        # yy = goal_y - self.y

        # orientation = Vector(*self.velocity).angle((xx,yy))

        area = (self.x - 80, self.y - 80, self.x + 80, self.y + 80)
        img = PILImage.open("./images/mask.png").convert('L')
        cropped_image = img.rotate(90, expand=True).crop(area)
        #car_image = PILImage.open("./images/car.png").resize((20,20))
        #cropped_image.paste(car_image.rotate(orientation, expand=True),(150,155))
        #cropped_image.save("./images/cropped_image.jpg")
        cropped_image.thumbnail((28,28))

        # x_coord1 = int(self.x)-40 if self.x > 40 else 0
        # x_coord2 = int(self.x)+40 if self.x < 1389 else 1419
        # y_coord1 = int(self.y)-40 if self.y > 40 else 0
        # y_coord2 = int(self.y)+40 if self.y < 620 else 650

        # cropped_array = sand[x_coord1:x_coord2, y_coord1:y_coord2]
        # cropped_image = PILImage.fromarray(cropped_array.astype("uint8")*255)
        # cropped_image.thumbnail((32,32))
        
        return cropped_image


class Game(Widget):

    car = ObjectProperty(None)

    reward_window = []

    start_timesteps = 1e4 # Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
    #eval_freq = 300 # How often the evaluation step is performed (after how many timesteps)
    max_timesteps = 5e5 # Total number of iterations/timesteps
    #max_episodes = 200 #int(1e8)  # max num of episodes
    eval_episode = 1
    
    expl_noise = 0.1 # Exploration noise - STD value of exploration Gaussian noise
    batch_size = 2 # Size of the batch
    discount = 0.99 # Discount factor gamma, used in the calculation of the total discounted reward
    tau = 0.005 # Target network update rate
    policy_noise = 0.2 # STD of Gaussian noise added to the actions for the exploration purposes
    noise_clip = 0.5 # Maximum value of the Gaussian noise added to the actions (policy)
    policy_freq = 2 # Number of iterations to wait before the policy network (Actor model) is updated

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_timesteps = 0
    done = True
    running_score = 0

    done_counter = 0

    """ parameters for epsilon decay """
    # epsilon_start = 1
    # epsilon_final = 0.01
    # decay_rate = max_episodes / 50


    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(9, 0)

    
    def get_orientation(self):
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))
        return torch.FloatTensor(np.array(orientation))


    def reset(self):
        # xx = goal_x - self.car.x
        # yy = goal_y - self.car.y
        # orientation = Vector(*self.car.velocity).angle((xx,yy))

        self.car.x = np.random.randint(200, 1400) #np.random.uniform(200, 1400, (1,))
        self.car.y = np.random.randint(200, 600)

        area = (self.car.x - 80, self.car.y - 80, self.car.x + 80, self.car.y + 80)

        img = PILImage.open("./images/mask.png").convert('L')
        cropped_image = img.rotate(90, expand=True).crop(area)
        #car_image = PILImage.open("./images/car.png").resize((20,20))
        #cropped_image.paste(car_image.rotate(orientation, expand=True),(150,155))
        #cropped_image.save("./images/cropped_image.jpg")
        cropped_image.thumbnail((32,32))
 
        # x_coord1 = int(self.x)-40 if self.x > 40 else 0
        # x_coord2 = int(self.x)+40 if self.x < 1389 else 1419
        # y_coord1 = int(self.y)-40 if self.y > 40 else 0
        # y_coord2 = int(self.y)+40 if self.y < 620 else 650

        # cropped_array = sand[x_coord1:x_coord2, y_coord1:y_coord2]

        # cropped_image = PILImage.fromarray(cropped_array.astype("uint8")*255)
        # cropped_image.thumbnail((32,32))

        self.stack = [np.expand_dims(cropped_image, axis=0)]

        return torch.FloatTensor(self.stack).permute(1, 0, 2, 3)


    def eval_reset(self):
        # xx = goal_x - self.car.x
        # yy = goal_y - self.car.y
        # orientation = Vector(*self.car.velocity).angle((xx,yy))

        # self.car.x = np.random.randint(200, 1400) #np.random.uniform(200, 1400, (1,))
        # self.car.y = np.random.randint(200, 600)

        area = (self.car.x - 80, self.car.y - 80, self.car.x + 80, self.car.y + 80)

        img = PILImage.open("./images/mask.png").convert('L')
        cropped_image = img.rotate(90, expand=True).crop(area)
        #car_image = PILImage.open("./images/car.png").resize((20,20))
        #cropped_image.paste(car_image.rotate(orientation, expand=True),(150,155))
        #cropped_image.save("./images/cropped_image.jpg")
        cropped_image.thumbnail((32,32))
 
        # x_coord1 = int(self.x)-40 if self.x > 40 else 0
        # x_coord2 = int(self.x)+40 if self.x < 1389 else 1419
        # y_coord1 = int(self.y)-40 if self.y > 40 else 0
        # y_coord2 = int(self.y)+40 if self.y < 620 else 650

        # cropped_array = sand[x_coord1:x_coord2, y_coord1:y_coord2]

        # cropped_image = PILImage.fromarray(cropped_array.astype("uint8")*255)
        # cropped_image.thumbnail((32,32))

        self.stack = [np.expand_dims(cropped_image, axis=0)]

        return torch.FloatTensor(self.stack).permute(1, 0, 2, 3)


    def step(self, action):
        global goal_x
        global goal_y
        global distance

        cropped_image = self.car.move(action.tolist()[0])

        self.stack.pop(0)
        self.stack.append(np.expand_dims(cropped_image, axis=0))

        next_state = torch.FloatTensor(self.stack).permute(1, 0, 2, 3)

        new_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        new_orientation = self.get_orientation()

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            #print(1, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            reward = -0.5
            self.done = False
        else: # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            reward = 20
            self.done = False
            #print(0, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
            if new_distance < distance:
                reward = 25
                self.done = False

        if self.car.x < 5:
            self.car.x = 5
            reward = -2000
            self.done = True
        if self.car.x > self.width - 20:
            self.car.x = self.width - 20
            reward = -2000
            self.done = True
        if self.car.y < 5:
            self.car.y = 5
            reward = -2000
            self.done = True
        if self.car.y > self.height - 20:
            self.car.y = self.height - 20
            reward = -2000
            self.done = True

        if self.car.x < 20:
            self.car.x = 20
            reward = -2000
            self.done = True
        if self.car.x > self.width - 20:
            self.car.x = self.width - 20
            reward = -2000
            self.done = True
        if self.car.y < 20:
            self.car.y = 20
            reward = -2000
            self.done = True
        if self.car.y > self.height - 20:
            self.car.y = self.height - 20
            reward = -2000
            self.done = True


        if new_distance < 25:
            if swap == 1:
                goal_x = 1300
                goal_y = 500
                swap = 0
                reward = 200
                self.done = True
            else:
                goal_x = 30
                goal_y = 8
                swap = 1
                reward = 200
                self.done = True
        distance = new_distance
        return next_state, new_distance, new_orientation, reward, self.done


    def evaluate_policy(self, brain, distance, orientation, eval_episode):
        avg_reward = 0.
        #print('in eval distance is ', distance)
        if eval_episode != 0 and eval_episode < _max_eval_episodes:
            state = self.reset()
            done = False
            if not done:
                action = brain.select_action(state, torch.FloatTensor(np.array(distance)), orientation)
                state, distance, orientation, reward, done = self.step(action)
                avg_reward += reward
                avg_reward /= eval_episode
            print ("---------------------------------------")
            print ("Average Reward over the Evaluation Step: %f" % (avg_reward))
            print ("---------------------------------------")
            return avg_reward


    # def score(self):
    #     return sum(self.reward_window)/(len(self.reward_window)+1.)

    def evaluate(self):
        brain.load(file_name, './pytorch_models/')


    def eval_update(self, dt):
        global brain
        global reward
        global scores
        global distance
        global orientation
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap

        global _max_episode_steps
        global _max_eval_episodes
        global episode_num
        global episode_reward
        global state
        global evaluations
        goal_x = 1300
        goal_y = 500
        #episode_reward = 0
        #episode_num = 0

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            orientation = self.get_orientation()
            distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
            evaluations = [] #[self.evaluate_policy(brain, distance, orientation, self.eval_episode)]

        if self.total_timesteps < self.max_timesteps:
            #print('total_timesteps is ', self.total_timesteps)
            #print('done is ', self.done)
            if self.done:
                self.done_counter += 1
                if self.total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, episode_num, episode_reward))
                    #brain.train(replay_buffer, self.episode_timesteps)

                # We evaluate the episode and we save the policy
                # if self.timesteps_since_eval >= self.eval_freq:
                #     self.timesteps_since_eval %= self.eval_freq
                #     evaluations.append(self.evaluate_policy(brain, distance, orientation, self.eval_episode))
                #     print('evaluations is ', evaluations)
                #     brain.save(file_name, directory="./pytorch_models")
                #     np.save("./results/%s" % (file_name), evaluations)

                state = self.eval_reset()
                #print('AFTER TRAIN')

                # Set the Done to False
                self.done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                self.episode_timesteps = 0
                episode_num += 1
                self.eval_episode += 1

            # Before 20000 timesteps, we play random actions
            if self.total_timesteps < self.start_timesteps:
                action = np.random.uniform(-5, 5, (1,))
            else:
                orientation = self.get_orientation()
                action = brain.select_action(state, torch.FloatTensor(np.array(distance)), orientation)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if self.expl_noise != 0:
                    #expl_noise = (self.epsilon_start - self.epsilon_final) * math.exp(-1. * self.total_timesteps / self.decay_rate)
                    action = (action + np.random.normal(0, self.expl_noise, size=action_dim)).clip(-5, 5)


            scores.append(brain.score())

            cropped_image = self.car.move(action.tolist()[0])

            self.stack.pop(0)
            self.stack.append(np.expand_dims(cropped_image, axis=0))

            next_state = torch.FloatTensor(self.stack).permute(1, 0, 2, 3)

            new_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

            new_orientation = self.get_orientation()

            if sand[int(self.car.x),int(self.car.y)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                #print(1, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                reward = -0.5
                self.done = False
            else: # otherwise
                self.car.velocity = Vector(2, 0).rotate(self.car.angle)
                reward = 20
                self.done = False
                #print(0, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                if new_distance < distance:
                    reward = 25
                    self.done = False

            if self.car.x < 5:
                self.car.x = 5
                reward = -2000
                self.done = True
            if self.car.x > self.width - 20:
                self.car.x = self.width - 20
                reward = -2000
                self.done = True
            if self.car.y < 5:
                self.car.y = 5
                reward = -2000
                self.done = True
            if self.car.y > self.height - 20:
                self.car.y = self.height - 20
                reward = -2000
                self.done = True

            if self.car.x < 20:
                self.car.x = 20
                reward = -2000
                self.done = True
            if self.car.x > self.width - 20:
                self.car.x = self.width - 20
                reward = -2000
                self.done = True
            if self.car.y < 20:
                self.car.y = 20
                reward = -2000
                self.done = True
            if self.car.y > self.height - 20:
                self.car.y = self.height - 20
                reward = -2000
                self.done = True


            if new_distance < 25:
                if swap == 1:
                    goal_x = 1300
                    goal_y = 500
                    swap = 0
                    reward = 200
                    self.done = True
                else:
                    goal_x = 30
                    goal_y = 8
                    swap = 1
                    reward = 200
                    self.done = True
            distance = new_distance

            # We check if the episode is done
            done_bool = 1 if self.episode_timesteps + 1 == _max_episode_steps else float(self.done)
            if self.episode_timesteps + 1 == _max_episode_steps:
                self.done = True
  
            # We increase the total reward
            episode_reward += reward
  
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add(state, next_state, distance, new_distance, orientation, new_orientation, action, reward, done_bool)
            #replay_buffer.add(state, next_state, new_distance, action, reward, done_bool)

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            state = next_state
            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1


    def update(self, dt):

        global brain
        global reward
        global scores
        global distance
        global orientation
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap

        global _max_episode_steps
        global _max_eval_episodes
        global episode_num
        global episode_reward
        global state
        global evaluations
        goal_x = 1300
        goal_y = 500
        #episode_reward = 0
        #episode_num = 0

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            orientation = self.get_orientation()
            distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
            evaluations = [] #[self.evaluate_policy(brain, distance, orientation, self.eval_episode)]

        if self.total_timesteps < self.max_timesteps:
            #print('total_timesteps is ', self.total_timesteps)
            #print('done is ', self.done)
            if self.done:
                self.done_counter += 1
                if self.total_timesteps != 0:
                    print("Total Timesteps: {} Episode Num: {} Reward: {}".format(self.total_timesteps, episode_num, episode_reward))
                    brain.train(replay_buffer, self.episode_timesteps)

                # We evaluate the episode and we save the policy
                # if self.timesteps_since_eval >= self.eval_freq:
                #     self.timesteps_since_eval %= self.eval_freq
                #     evaluations.append(self.evaluate_policy(brain, distance, orientation, self.eval_episode))
                #     print('evaluations is ', evaluations)
                #     brain.save(file_name, directory="./pytorch_models")
                #     np.save("./results/%s" % (file_name), evaluations)

                state = self.reset()
                #print('AFTER TRAIN')

                # Set the Done to False
                self.done = False

                # Set rewards and episode timesteps to zero
                episode_reward = 0
                self.episode_timesteps = 0
                episode_num += 1
                self.eval_episode += 1

            # Before 20000 timesteps, we play random actions
            if self.total_timesteps < self.start_timesteps:
                action = np.random.uniform(-5, 5, (1,))
            else:
                orientation = self.get_orientation()
                action = brain.select_action(state, torch.FloatTensor(np.array(distance)), orientation)
                # If the explore_noise parameter is not 0, we add noise to the action and we clip it
                if self.expl_noise != 0:
                    #expl_noise = (self.epsilon_start - self.epsilon_final) * math.exp(-1. * self.total_timesteps / self.decay_rate)
                    action = (action + np.random.normal(0, self.expl_noise, size=action_dim)).clip(-5, 5)


            scores.append(brain.score())

            cropped_image = self.car.move(action.tolist()[0])

            self.stack.pop(0)
            self.stack.append(np.expand_dims(cropped_image, axis=0))

            next_state = torch.FloatTensor(self.stack).permute(1, 0, 2, 3)

            new_distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

            new_orientation = self.get_orientation()

            if sand[int(self.car.x),int(self.car.y)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                #print(1, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                reward = -0.5
                self.done = False
            else: # otherwise
                self.car.velocity = Vector(9, 0).rotate(self.car.angle)
                reward = 20
                self.done = False
                #print(0, goal_x, goal_y, new_distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                if new_distance < distance:
                    reward = 25
                    self.done = False

            if self.car.x < 5:
                self.car.x = 5
                reward = -2000
                self.done = True
            if self.car.x > self.width - 20:
                self.car.x = self.width - 20
                reward = -2000
                self.done = True
            if self.car.y < 5:
                self.car.y = 5
                reward = -2000
                self.done = True
            if self.car.y > self.height - 20:
                self.car.y = self.height - 20
                reward = -2000
                self.done = True

            if self.car.x < 20:
                self.car.x = 20
                reward = -2000
                self.done = True
            if self.car.x > self.width - 20:
                self.car.x = self.width - 20
                reward = -2000
                self.done = True
            if self.car.y < 20:
                self.car.y = 20
                reward = -2000
                self.done = True
            if self.car.y > self.height - 20:
                self.car.y = self.height - 20
                reward = -2000
                self.done = True


            if new_distance < 25:
                if swap == 1:
                    goal_x = 1300
                    goal_y = 500
                    swap = 0
                    reward = 200
                    self.done = True
                else:
                    goal_x = 30
                    goal_y = 8
                    swap = 1
                    reward = 200
                    self.done = True
            distance = new_distance

            # We increase the total reward
            episode_reward += reward

            # We check if the episode is done
            done_bool = 1 if self.episode_timesteps + 1 == _max_episode_steps else float(self.done)
            if self.episode_timesteps + 1 == _max_episode_steps:
                self.done = True
  
            if self.done:
                if episode_reward > 0: #and self.done_counter % 3 == 0:
                # We add the last policy evaluation to our list of evaluations and we save our model
                #evaluations.append(evaluate_policy(brain, distance, orientation, self.eval_episode))
                    if save_models: brain.save("%s_%s_%s" % (file_name, int(episode_reward), self.done_counter), directory="./pytorch_models")
                #np.save("./results/%s" % (file_name), evaluations)
  
            # We store the new transition into the Experience Replay memory (ReplayBuffer)
            replay_buffer.add(state, next_state, distance, new_distance, orientation, new_orientation, action, reward, done_bool)
            #replay_buffer.add(state, next_state, new_distance, action, reward, done_bool)

            # We update the state, the episode timestep, the total timesteps, and the timesteps since the evaluation of the policy
            state = next_state
            self.episode_timesteps += 1
            self.total_timesteps += 1
            self.timesteps_since_eval += 1



# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 20.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 20)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 20 : int(touch.x) + 20, int(touch.y) - 20 : int(touch.y) + 20] = 1

            
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        #For Training
        #Clock.schedule_interval(parent.update, 1.0/60.0)
        #For Inferencing
        parent.evaluate()
        Clock.schedule_interval(parent.eval_update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()